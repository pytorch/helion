"""Render the selected paged-attention schedule into an ordinary root."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .autotune.mma_dag_surface import carries_capture_overlap
from .cute_mma_dag import AXIS_ROOT
from .cute_mma_dag import MMA_DAG_KEY_CLIP_OFF
from .cute_mma_dag import MMA_DAG_PARTITION_BALANCED
from .cute_mma_dag import MMA_DAG_PARTITION_ITEM
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from .cute_mma_dag_codegen import emit_mma_dag_module_statements
from .cute_mma_dag_codegen import mma_dag_alias_plan
from .cute_mma_dag_emitter_base import MmaDagEmitterBase
from .cute_mma_dag_emitter_base import _Evaluator
from .fx_matcher import _is_call
from .packet_flush_names import PAIRED_BODY_NAME
from .packet_flush_names import PAIRED_BODY_STORE_POLICY
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_REVERSED
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_SOURCE
from .paged_attention_partition import paired_stream_page_bound
from .paged_attention_partition import partition_workspace_arg
from .paged_attention_prefetch import BOUNDARY_PREFETCH_SIBLING
from .paged_attention_prefetch import boundary_prefetch_form
from .paged_attention_prefetch import boundary_prefetch_layout
from .paged_attention_registers import PAIRED_DEFAULT_REGISTER_QUOTAS
from .paged_attention_release import MMA_DAG_RELEASE_LEAD_DEFAULT
from .paged_attention_shared_memory import PAIRED_DEFAULT_VALUE_RING
from .paged_attention_transport import paged_attention_launch_warps
from .paged_attention_transport import paged_attention_tma_accesses
from .passes.pdl import exit_release_applies
from .passes.pdl import loader_tail_release_applies
from .passes.pdl import pdl_launch_ctas
from .passes.pdl import pdl_resident_capacity
from .pdl_release import capture_policy_available
from .state import cute_codegen_state
from .tile_ops import cute_tile_begin_expr

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .cute_mma_dag import Access
    from .cute_mma_dag import MmaDagPlan
    from .paged_attention_transport import PagedAttentionSchedule


class _UniformPrelude(MmaDagEmitterBase):
    """The shared scalar evaluator without a fragment or shared-tile layout."""

    def __init__(self, cg: GenerateAST, plan: MmaDagPlan) -> None:
        self.cg = cg
        self.df = cg.device_function
        self.env = CompileEnvironment.current()
        self.plan = plan
        self.program = plan.program
        self.lines = []
        self.indent = ""
        self.accesses = {access.node: access for access in plan.program.accesses}
        self.base_vars = {}
        self.iter_vars = {}
        self.scalar_vars = {}
        self.values = {}
        self.name_depth = {}
        self.redirect = None
        self.hoists = []
        self.ring_aliases = {}
        self.ring_ready = set()
        self.pairs = False
        self.packable_memo = {}
        self.fold_addresses = False
        self.evaluator = _Evaluator(self)


def _unit_bounds_helper(
    cg: GenerateAST,
    plan: MmaDagPlan,
    lines: list[str],
    values: tuple[str, str, str, str],
    *,
    live: bool = False,
) -> tuple[str, str | None, list[str]]:
    """Wrap the scalar prelude as module functions of a unit's tile offset.

    A persistent CTA evaluates a unit's item, loop bounds and length once
    per unit and role instead of once at kernel entry. ``lines`` are the
    evaluator's own statements, so every integer operation keeps its source
    width and order; only the leading tile offset becomes the parameter.

    ``live`` keeps only the statements the four results depend on and makes
    no second function: a role that evaluates another unit beside the
    kernel's own prelude (the boundary prefetch of a one-unit CTA) then
    names the tensors the bounds are read from and no other argument.

    The prelude's global load is the unit's length, which the CTA's plan
    scan reads for every unit. When every load of the prelude reads that
    one Int32 word, a second function takes the loaded value as its second
    parameter and is the first one otherwise, statement for statement: a
    role then evaluates a unit from the length the scan kept instead of
    loading it again.

    Return the first function's name, the second's (None without one) and
    the kernel arguments both take last: passing them from the kernel also
    keeps the ones only the prelude reads.
    """
    df = cg.device_function
    program = plan.program
    (lead,) = program.lead_axes
    unit = cute_tile_begin_expr(cg, lead.block_id)
    assert unit.isidentifier()
    # The row and dimension axes have one tile: their offsets are zero.
    single_tiles = []
    for axis in program.axes:
        if axis.kind == AXIS_ROOT:
            assert plan.block(axis) >= axis.extent
            offset = cute_tile_begin_expr(cg, axis.block_id)
            assert offset.isidentifier()
            single_tiles.append(offset)
    result = (
        f"(cutlass.Int32({values[0]}), cutlass.Int64({values[1]}), "
        f"cutlass.Int64({values[2]}), cutlass.Int32({values[3]}))"
    )
    tree = ast.parse("\n".join([*lines, result]))
    # Each line's target and value, in order.
    defined: dict[str, ast.expr] = {}
    for statement in tree.body[:-1]:
        # Straight-line scalar code: the CuTe tracer runs it as plain Python.
        assert isinstance(statement, ast.Assign) and len(statement.targets) == 1
        (target,) = statement.targets
        assert isinstance(target, ast.Name)
        defined[target.id] = statement.value
    assert len(defined) == len(lines)
    if live:
        # Straight-line single assignments: one backward pass finds what the
        # results read, through every statement they depend on.
        needed = {
            node.id for node in ast.walk(tree.body[-1]) if isinstance(node, ast.Name)
        }
        for target in reversed(list(defined)):
            if target in needed:
                needed |= {
                    node.id
                    for node in ast.walk(defined[target])
                    if isinstance(node, ast.Name)
                }
        lines = [
            line
            for line, target in zip(lines, defined, strict=True)
            if target in needed
        ]
        defined = {
            target: value for target, value in defined.items() if target in needed
        }
        tree = ast.parse("\n".join([*lines, result]))
    free = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - set(defined)
    parameters = [argument.name for argument in df.arguments if argument.name in free]
    name = df.new_var("_helion_pa_unit_bounds")
    body = [f"{offset} = 0" for offset in single_tiles if offset in free]
    cg.ensure_module_statements(
        f"def {name}({', '.join([unit, *parameters])}):\n"
        + "".join(f"    {line}\n" for line in [*body, *lines])
        + f"    return {result}\n"
    )
    # A unit's length is the prelude's global load: one Int32 word, possibly
    # read more than once. A second function then takes that word as its
    # second parameter in place of every load. A persistent partition's
    # inputs do not change while the kernel runs, so one read stands for all.
    loads = [
        target
        for target, value in defined.items()
        if any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func).rpartition(".")[2].startswith("ldg_")
            for node in ast.walk(value)
        )
    ]
    addresses = set()
    for target in loads:
        value = defined[target]
        address = None
        if (
            isinstance(value, ast.Call)
            and ast.unparse(value.func).rpartition(".")[2] == "ldg_s32"
            and len(value.args) == 1
            and not value.keywords
        ):
            (argument,) = value.args
            if isinstance(argument, ast.Name):
                address = ast.unparse(defined[argument.id])
        addresses.add(address)
    scanned_name = None
    if (
        not live
        and values[3] in loads
        and len(addresses) == 1
        and None not in addresses
    ):
        scanned = df.new_var("scanned_length")
        scanned_lines = list(lines)
        order = list(defined)
        for target in loads:
            scanned_lines[order.index(target)] = f"{target} = cutlass.Int32({scanned})"
        scanned_name = df.new_var("_helion_pa_unit_bounds_scanned")
        cg.ensure_module_statements(
            f"def {scanned_name}({', '.join([unit, scanned, *parameters])}):\n"
            + "".join(f"    {line}\n" for line in [*body, *scanned_lines])
            + f"    return {result}\n"
        )
    return name, scanned_name, parameters


def codegen_paged_attention(cg: GenerateAST, schedule: PagedAttentionSchedule) -> None:
    """Keep the generic grid/arguments and replace the proved root body."""
    df = cg.device_function
    selected = df.cute_state.root_plans.mma_dag_plan
    assert selected is not None and selected.attention is schedule
    transport = schedule.transport
    semantic = transport.semantic
    assert schedule.launch_warps == paged_attention_launch_warps(
        per_key=semantic.denominator_per_key,
        scoped_registers=schedule.scoped_registers,
        paired_items=schedule.paired_items,
    )
    emit_mma_dag_module_statements(cg)
    cg.ensure_module_statements(
        "import helion._compiler.cute._mma_dag_tcgen05_runtime as _helion_mma_dag_tc\n"
    )

    if schedule.paired_items:
        assert semantic.denominator_per_key and not schedule.scoped_registers
        assert schedule.tmem_columns == 512 and schedule.min_blocks_per_mp == 1
        cg.ensure_module_statements(
            "import helion._compiler.cute._paged_attention_paired_pipeline as _helion_paged_paired\n"
        )
        body_name = f"_helion_paged_paired.{PAIRED_BODY_NAME}"
    elif semantic.denominator_per_key:
        assert schedule.query_columns in (8, 32, 40)
        assert 0 < transport.query_rows <= schedule.query_columns
        cg.ensure_module_statements(
            "import helion._compiler.cute._paged_attention_partial_pipeline as _helion_paged_partial\n"
        )
        body_name = "_helion_paged_partial.paged_attention_partial_body"
    else:
        assert not schedule.scoped_registers
        assert schedule.query_columns == 64
        cg.ensure_module_statements(
            "import helion._compiler.cute._paged_attention_pipeline as _helion_paged_attention\n"
        )
        body_name = "_helion_paged_attention.paged_attention_body"

    # Reuse the ordinary uniform expression evaluator. In particular, tensor
    # INT32/INT64 operations finish before range induction widens them; a
    # length/window formula reconstructed here could change overflow order.
    emitter = _UniformPrelude(cg, selected)
    emitter.define_axis_bases()
    emitter.define_bases()
    begin, end, _static_trip = emitter.loop_bounds(semantic.loop)
    length = emitter.scalar_vars[semantic.coordinates.lengths.node]
    item = emitter.uniform(transport.item, {})

    def tensor_name(access: Access) -> str:
        name = df.tensor_arg(access.fake, prefer_name=access.name).name
        df.placeholder_args.add(name)
        return name

    accesses = selected.program.accesses

    @functools.cache
    def program_tensors(operation: object) -> frozenset[str]:
        """The tensors the program loads (``memory_ops.load``) or stores
        (``memory_ops.store``), by name: computed once per operation, where
        the render first asks, since naming a tensor binds its argument."""
        return frozenset(
            tensor_name(access)
            for access in accesses
            if _is_call(access.node, operation)
        )

    def stores_a_loaded_tensor() -> bool:
        """Whether the program stores a tensor it also loads. A prefetch
        role reads the page table where a launch may read its inputs, ahead
        of the dependency wait of a captured launch, whose proof covers the
        tensors the program loads: such a program gets no role."""
        return bool(
            program_tensors(memory_ops.load) & program_tensors(memory_ops.store)
        )

    maps = tuple(df.new_var("pa_map") for _ in range(4))
    wrapper: dict[str, object] = {"kind": "helion_paged_attention"}
    for index, (access, param) in enumerate(
        zip(
            paged_attention_tma_accesses(semantic),
            maps,
            strict=True,
        )
    ):
        wrapper[f"tma{index}_name"] = tensor_name(access)
        wrapper[f"tma{index}_param"] = param
        df.wrapper_only_params.append(param)
        df.wrapper_only_param_annotations[param] = (
            "cutlass.GridConstant[_helion_mma_dag_tc.cuda.TensorMap]"
        )
    query_arg = ""
    if schedule.query_tma is not None:
        query_param = df.new_var("pa_query_map")
        query_tma = schedule.query_tma
        wrapper["tma4_name"] = tensor_name(semantic.query)
        wrapper["query_tma_param"] = query_param
        wrapper["query_tma_layout"] = (
            query_tma.shape,
            query_tma.strides,
            query_tma.box,
        )
        df.wrapper_only_params.append(query_param)
        df.wrapper_only_param_annotations[query_param] = (
            "cutlass.GridConstant[_helion_mma_dag_tc.cuda.TensorMap]"
        )
        query_arg = f", query_map={query_param}"
    # The occupancy this body launches with; the dependent-launch release
    # rule sizes the grid's waves from it (``passes.pdl``).
    df.cute_state.tcgen05.min_blocks_per_mp = schedule.min_blocks_per_mp
    if schedule.partition != MMA_DAG_PARTITION_ITEM:
        # The launch is one wave of persistent CTAs, at most one per SM
        # (``paged_attention_partition.partition_launch_grid``). That count is
        # what the dependent-launch release rules size the waves from, so it
        # is set ahead of the tail-release test below.
        num_sm = emitter.env.config_spec.num_sm
        assert num_sm > 1
        df.cute_state.lanes.attention_partition = schedule.partition
        df.cute_state.lanes.launch_ctas = min(schedule.partition_units, num_sm)
    # The config asks for the overlap and the rest of it permits that
    # (``carries_capture_overlap``, by the rule normalization applies); the
    # checks below are this emitter's own: one root, and the effect proof.
    available = capture_policy_available(CompileEnvironment.current().config_spec)
    guarded_pdl = (
        carries_capture_overlap(available, df.config.config)
        and len(HostFunction.current().device_ir.root_ids) == 1
    )
    assert not guarded_pdl or schedule.paired_items
    if guarded_pdl:
        expected_stores = {
            semantic.output.node,
            *(access.node for access in semantic.row_outputs),
        }
        actual_stores = {
            access.node
            for access in selected.program.accesses
            if _is_call(access.node, memory_ops.store)
        }
        guarded_pdl = actual_stores == expected_stores and all(
            _is_call(access.node, memory_ops.load)
            or _is_call(access.node, memory_ops.store)
            for access in selected.program.accesses
        )
    mode_arg = ""
    tail_release = False
    if guarded_pdl:
        mode = df.new_var("capture_pdl_wait_mode")
        df.wrapper_only_params.append(mode)
        df.wrapper_only_param_annotations[mode] = "cutlass.Int32"
        wrapper["capture_pdl_mode_param"] = mode
        # The semantic proof consumes the whole root. The selected body has
        # only the proved output and row global stores, and every other access is a
        # read (including the generated scalar prelude). Its existing alias
        # plan enumerates all of those argument storages for the runtime proof.
        # A balanced partition also reads and writes its launch workspace,
        # every access behind the dependency wait: the plan lists it as a
        # stored argument that no read ahead of the wait touches.
        wrapper["capture_pdl_contract"] = "paired-output-stores-v1"
        mode_arg = f", capture_pdl_wait_mode={mode}"
        if df.config.config.get(
            "cute_pdl_release"
        ) == "loader_tail" and loader_tail_release_applies(df):
            # Past one wave the automatic release takes over: the entry of
            # every CTA (``passes.pdl.prepare_device_body``).
            df.cute_state.lanes.pdl_loader_tail_release = True
            mode_arg += ", capture_release_tail=True"
            tail_release = True
            if schedule.release_lead != MMA_DAG_RELEASE_LEAD_DEFAULT:
                # The lead is the tail release's: a launch that releases
                # elsewhere (past one wave, a declined capture contract)
                # renders the kernel without it, as it does the release.
                mode_arg += f", capture_release_lead={schedule.release_lead}"
    coords = semantic.coordinates
    coordinate_dtype = "Int64" if coords.key_dtype == torch.int64 else "Int32"
    panel_arg = ""
    if any(schedule.tma_load_hints):
        panel_arg += f", tma_load_hints={schedule.tma_load_hints!r}"
    if semantic.denominator_per_key:
        panel_arg += f", query_columns={schedule.query_columns}"
        if not schedule.paired_items:
            panel_arg += f", scoped_registers={schedule.scoped_registers!r}"
    if schedule.paired_items and schedule.v_lag:
        panel_arg += f", v_lag={schedule.v_lag}"
    if schedule.paired_items and schedule.sleep_waits:
        panel_arg += ", sleep_waits=True"
    if schedule.paired_items and schedule.early_tmem_retirement:
        panel_arg += ", early_tmem_retirement=True"
        if exit_release_applies(df):
            # The generic exit release follows the body in every thread: keep
            # one common end so no finished role releases the dependents early.
            panel_arg += ", join_before_exit=True"
    if schedule.key_clip != MMA_DAG_KEY_CLIP_OFF:
        assert schedule.paired_items and schedule.fast_math
        assert not schedule.value_metadata_skip
        panel_arg += f", key_clip={schedule.key_clip!r}"
    if schedule.register_quotas != PAIRED_DEFAULT_REGISTER_QUOTAS:
        assert schedule.paired_items
        panel_arg += f", register_quotas={schedule.register_quotas!r}"
    if schedule.paired_items:
        page_bound = paired_stream_page_bound(
            semantic.page_table.fake.shape[1],
            schedule.partition_units
            if schedule.partition != MMA_DAG_PARTITION_ITEM
            else 1,
        )
        if page_bound:
            # The body's per-tile loops count in Int32 below this bound.
            panel_arg += f", stream_page_bound={page_bound}"
    if schedule.value_ring != PAIRED_DEFAULT_VALUE_RING:
        # The body repeats the contract's check on the table it allocates,
        # against the capacity the depth was admitted on.
        assert schedule.paired_items and not schedule.value_metadata_skip
        panel_arg += (
            f", value_ring={schedule.value_ring}"
            f", shared_memory_capacity={schedule.smem_capacity}"
        )
    if schedule.value_ring_tiles:
        # A persistent CTA's ring counted in tiles, in the place of the
        # default depth in pages; the body repeats the contract's check on
        # the table it allocates, as for a deeper ring in pages.
        assert schedule.paired_items and not schedule.value_metadata_skip
        assert schedule.partition != MMA_DAG_PARTITION_ITEM
        assert schedule.value_ring == PAIRED_DEFAULT_VALUE_RING
        assert semantic.mask.window_left is not None
        panel_arg += (
            f", value_ring_tiles={schedule.value_ring_tiles}"
            f", shared_memory_capacity={schedule.smem_capacity}"
        )
    if schedule.page_order != MMA_DAG_PAGE_ORDER_SOURCE:
        # Another order of a unit's tiles: fast math. The bucketed order
        # permutes whole interior tiles on the control warp the
        # value-metadata actor and a persistent partition would take, so it
        # has one unit per CTA; the reversed walk permutes the source's
        # iterations and has no role, on any partition.
        assert schedule.paired_items and schedule.fast_math
        assert not schedule.value_metadata_skip
        panel_arg += f", page_order={schedule.page_order!r}"
        if schedule.page_order == MMA_DAG_PAGE_ORDER_REVERSED:
            # The walk reflects a loop the source does not, under a window.
            assert semantic.mask.window_left is not None
            assert semantic.coordinates.reflected is None
            if schedule.clipped_page_lead:
                panel_arg += f", clipped_page_lead={schedule.clipped_page_lead}"
        else:
            assert schedule.partition == MMA_DAG_PARTITION_ITEM
            panel_arg += f", page_order_shift={schedule.page_order_shift}"
    else:
        assert not schedule.clipped_page_lead
    store_policy = df.config.config.get("cute_store_policy", "default")
    if schedule.paired_items and store_policy != "default":
        # The policy of the kernel's output stores, which this body makes
        # itself: its O stores take the streaming cache operator or the L2
        # evict-last hint, with the values, addresses, count and order of
        # the plain ones. The hint's policy word is made at run time, so a
        # config on this schedule keeps ``l2_evict_last`` on every target
        # (``search_surface._normalize_paired_store_policy``). The value is
        # the policy's name: normalization replaces whatever compared equal
        # to one (``search_surface._normalize_cute_store_policy``).
        assert type(store_policy) is str
        panel_arg += f", {PAIRED_BODY_STORE_POLICY}={store_policy!r}"
    if schedule.probability_schedule != MMA_DAG_PROBABILITY_SCHEDULE_SERIAL:
        assert schedule.paired_items and semantic.score_reference is not None
        panel_arg += f", probability_schedule={schedule.probability_schedule!r}"
    if semantic.score_reference is not None:
        assert semantic.denominator_per_key
        panel_arg += ", reference_scores=True"
    if semantic.score_base2:
        panel_arg += ", base2_scores=True"
    if semantic.rowwise_rescale is not None:
        assert semantic.score_reference is not None and semantic.score_base2
        panel_arg += f", rowwise_rescale_threshold={semantic.rowwise_rescale.threshold_for(df.config)!r}"
        if semantic.rowwise_rescale.headwide:
            panel_arg += ", headwide_rescale=True"
    if semantic.coefficients_capped:
        assert schedule.paired_items and semantic.partial_statistics is not None
        panel_arg += ", coefficients_capped=True"
    if semantic.partial_statistics is not None:
        maximum, denominator = semantic.partial_statistics
        panel_arg += f", maximum_output={tensor_name(maximum)}"
        panel_arg += f", denominator_output={tensor_name(denominator)}"
    if coords.reflected is not None:
        assert semantic.denominator_per_key
        reflected_dtype = "Int64" if coords.reflected.dtype == torch.int64 else "Int32"
        panel_arg += f", reflected_dtype=cutlass.{reflected_dtype}"
    if semantic.output_reciprocal:
        panel_arg += ", output_reciprocal=True"
    for name, exponent in (
        ("key_post_scale_log2", semantic.key.post_scale_log2),
        ("value_post_scale_log2", semantic.value.post_scale_log2),
        ("output_scale_log2", semantic.output_scale_log2),
    ):
        if exponent:
            panel_arg += f", {name}={exponent}"
    if semantic.value.positive_zero:
        panel_arg += ", value_positive_zero=True"
    if schedule.value_metadata_skip:
        assert schedule.paired_items and semantic.value.positive_zero
        # Use the ordinary named argument, whose iterator includes its storage
        # offset. The normalized descriptor layout is in packed Int32 words.
        # Keeping this argument preserves the normal alias/capture read set.
        scales = tensor_name(semantic.value.scale)
        panel_arg += (
            ", value_metadata_scales=cute.make_tensor("
            f"cute.recast_ptr({scales}.iterator, dtype=cutlass.Int32), "
            f"cute.make_layout({transport.value.scale_shape!r}, "
            f"stride={transport.value.scale_strides!r}))"
        )
        if schedule.value_metadata_read_overlap:
            panel_arg += ", value_metadata_read_overlap=True"
    io_arg = ""
    if transport.direct_io is not None:
        assert semantic.denominator_per_key
        io = transport.direct_io
        io_arg = f", io_layouts={(io.query, io.output, io.lse)!r}"
        if io.partial_statistics is not None:
            io_arg += f", statistic_layouts={io.partial_statistics!r}"
    lse_arg = tensor_name(semantic.lse) if semantic.lse is not None else "None"
    work_item = (
        f"cutlass.Int32({item}), cutlass.Int64({begin}), cutlass.Int64({end}), "
        f"cutlass.Int32({length})"
    )
    partition_arg = ""
    workspace_name = None
    if schedule.partition != MMA_DAG_PARTITION_ITEM:
        # Every role evaluates the prelude per unit through the module
        # function; the kernel has no work item of its own. The generic grid
        # decode stays and is dead: the launch grid counts CTAs, not units.
        assert schedule.paired_items and schedule.query_tma is not None
        assert not schedule.value_metadata_skip
        helper, scanned_helper, parameters = _unit_bounds_helper(
            cg, selected, emitter.lines, (item, begin, end, length)
        )
        emitter.lines.clear()
        work_item = (
            "cutlass.Int32(0), cutlass.Int64(0), cutlass.Int64(0), cutlass.Int32(0)"
        )
        partition_arg = (
            f", partition={schedule.partition!r}, units={schedule.partition_units}"
            f", page_limit={schedule.partition_page_limit}"
            f", q_stages={schedule.q_stages}"
            f", slot_bits={schedule.partition_slot_bits}, unit_bounds={helper}"
            f", unit_bounds_args=({''.join(f'{name}, ' for name in parameters)})"
        )
        if scanned_helper is not None:
            partition_arg += f", unit_bounds_scanned={scanned_helper}"
        if schedule.partition == MMA_DAG_PARTITION_BALANCED:
            assert schedule.fast_math and semantic.partial_statistics is None
            workspace_name = partition_workspace_arg(
                cg, schedule.partition_record_bytes, emitter.env.config_spec.num_sm
            ).name
            partition_arg += (
                f", partition_workspace={workspace_name}"
                f", partition_record_bytes={schedule.partition_record_bytes}"
            )
            if schedule.partition_skew:
                partition_arg += f", partition_skew={schedule.partition_skew}"
        if schedule.boundary_prefetch_ahead:
            # The balanced partition's range-ahead prefetch: the spare warp
            # requests the first pages of the CTA a distance ahead. The
            # config contract admitted it from the bind's facts; the render
            # repeats what the role rests on and leaves it out where that
            # does not hold, as for the forms of one unit per CTA below.
            # The role reads the page table where the launch may read its
            # inputs, so it is left out where the program also stores a
            # tensor it loads (``stores_a_loaded_tensor``). It evaluates the
            # served unit from the lengths the plan scan kept, so it needs
            # the prelude's twin that takes a scanned length.
            assert schedule.partition == MMA_DAG_PARTITION_BALANCED
            assert not schedule.boundary_prefetch
            key_layout = boundary_prefetch_layout(transport)
            value_layout = boundary_prefetch_layout(transport, transport.value)
            aliased = stores_a_loaded_tensor()
            if (
                key_layout is not None
                and value_layout is not None
                and scanned_helper is not None
                and not aliased
                and len(HostFunction.current().device_ir.root_ids) == 1
            ):
                partition_arg += (
                    f", boundary_prefetch_ahead={schedule.boundary_prefetch_ahead}"
                    f", boundary_prefetch_key={tensor_name(transport.key.raw)}"
                    f", boundary_prefetch_key_scales={tensor_name(transport.key.scale)}"
                    f", boundary_prefetch_layout={key_layout!r}"
                    f", boundary_prefetch_value={tensor_name(transport.value.raw)}"
                    ", boundary_prefetch_value_scales="
                    f"{tensor_name(transport.value.scale)}"
                    f", boundary_prefetch_value_layout={value_layout!r}"
                )
    prefetch_arg = ""
    if schedule.boundary_prefetch:
        # The config contract admitted the form from the bind's facts; the
        # render repeats it against this launch and leaves the role out
        # where it does not hold. The grid is one CTA per unit on a known
        # capacity, and the form of a one-wave grid serves the units that
        # wait for the previous launch's SMs, which only a launch with the
        # capture contract and the release from its loader's tail has.
        # These are second gates. Normalization rejects every config that
        # fails one (``paged_attention_boundary_prefetch_reason``,
        # ``boundary_prefetch_policy_reason``), the paired schedule's launch
        # has the capacity and grid of the facts, and that schedule owns a
        # kernel's only root, so no config of today is decided here: a
        # config that reaches the render another way, or one whose final
        # effect proof declines the capture contract, gets the kernel
        # without the role.
        assert schedule.paired_items and not schedule.value_metadata_skip
        assert schedule.partition == MMA_DAG_PARTITION_ITEM
        prefetch_units = transport.items // 2
        prefetch_capacity = pdl_resident_capacity(df)
        prefetch_layout = boundary_prefetch_layout(transport)
        form = schedule.boundary_prefetch_form
        # The role reads the served unit's bounds and page ids where the
        # launch may read its inputs, ahead of the dependency wait of a
        # captured launch. That launch's proof covers the tensors the
        # program loads. The role names only such tensors (the page table,
        # the key operands and what the prelude's bounds read), so it is
        # left out where the program also stores one it loads.
        aliased = stores_a_loaded_tensor()
        if (
            prefetch_layout is not None
            and prefetch_capacity is not None
            and not aliased
            and len(HostFunction.current().device_ir.root_ids) == 1
            and pdl_launch_ctas(df) == prefetch_units
            and boundary_prefetch_form(prefetch_units, prefetch_capacity) == form
            and (form != BOUNDARY_PREFETCH_SIBLING or tail_release)
        ):
            # The kernel keeps its own prelude; the role evaluates the unit
            # it serves through the same statements as a function of the
            # unit.
            helper, _scanned, parameters = _unit_bounds_helper(
                cg, selected, emitter.lines, (item, begin, end, length), live=True
            )
            key_raw = tensor_name(transport.key.raw)
            key_scales = tensor_name(transport.key.scale)
            prefetch_arg = (
                f", boundary_prefetch={schedule.boundary_prefetch}"
                f", boundary_prefetch_form={form!r}"
                f", boundary_prefetch_capacity={prefetch_capacity}"
                f", boundary_prefetch_key={key_raw}"
                f", boundary_prefetch_key_scales={key_scales}"
                f", boundary_prefetch_layout={prefetch_layout!r}"
                f", units={prefetch_units}, unit_bounds={helper}"
                f", unit_bounds_args=({''.join(f'{name}, ' for name in parameters)})"
            )
            if form == BOUNDARY_PREFETCH_SIBLING:
                # The host installs, above the wait mode, how many of the
                # launch's first CTAs start beside the launch ahead.
                wrapper["capture_pdl_head_start"] = True
                if schedule.boundary_prefetch_cold:
                    # A launch without a head start serves too, to this
                    # depth: the role's own extension, rendered with it and
                    # left out with it.
                    prefetch_arg += (
                        f", boundary_prefetch_cold={schedule.boundary_prefetch_cold}"
                    )
    emitter.emit(
        f"{body_name}("
        f"{tensor_name(semantic.query)}, {tensor_name(semantic.page_table)}, "
        f"{tensor_name(semantic.output)}, {lse_arg}, "
        f"{work_item}, {coords.query_len}, {coords.group}, "
        f"{transport.query_rows}, {transport.heads}, {semantic.mask.window_left!r}, "
        f"cutlass.Float32({semantic.score_scale!r}), {schedule.fast_math!r}, "
        f"{', '.join(maps)}, coordinate_dtype=cutlass.{coordinate_dtype}{panel_arg}{io_arg}{query_arg}{mode_arg}{partition_arg}{prefetch_arg})"
    )
    for statement in ast.parse("\n".join(emitter.lines)).body:
        cg.add_statement(statement)
    wrappers = cute_codegen_state(cg).wrapper_plans
    wrappers.append(wrapper)
    alias_plan = mma_dag_alias_plan(df, selected.program)
    if workspace_name is not None:
        # The kernel stores to the workspace and, behind the dependency wait
        # only, loads from it: no other argument may overlap it, and a
        # captured launch may still read its inputs ahead of that wait.
        cast("list[str]", alias_plan["tensor_names"]).append(workspace_name)
        cast("list[bool]", alias_plan["stored"]).append(True)
    if guarded_pdl:
        loaded = program_tensors(memory_ops.load)
        alias_plan["capture_loaded"] = [
            name in loaded for name in cast("list[str]", alias_plan["tensor_names"])
        ]
    wrappers.append(alias_plan)
    df.cute_state.lanes.owned_root_block_dims = (schedule.launch_warps * 32, 1, 1)
