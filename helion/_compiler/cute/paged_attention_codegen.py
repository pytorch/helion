"""Render the selected paged-attention schedule into an ordinary root."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from .cute_mma_dag_codegen import emit_mma_dag_module_statements
from .cute_mma_dag_codegen import mma_dag_alias_plan
from .cute_mma_dag_emitter_base import MmaDagEmitterBase
from .cute_mma_dag_emitter_base import _Evaluator
from .fx_matcher import _is_call
from .paged_attention_transport import paged_attention_launch_warps
from .paged_attention_transport import paged_attention_tma_accesses
from .pdl_release import CUTE_PDL_CAPTURE_RELEASE_CHOICES
from .pdl_release import CUTE_PDL_RELEASE_DEFAULT
from .state import cute_codegen_state

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
        body_name = "_helion_paged_paired.paged_attention_paired_body"
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
    guarded_pdl = (
        schedule.paired_items
        and df.config.config.get("cute_pdl_capture_overlap") is True
        and df.config.config.get("cute_pdl") is True
        and df.config.config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
        in CUTE_PDL_CAPTURE_RELEASE_CHOICES
        and len(HostFunction.current().device_ir.root_ids) == 1
    )
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
    if guarded_pdl:
        mode = df.new_var("capture_pdl_wait_mode")
        df.wrapper_only_params.append(mode)
        df.wrapper_only_param_annotations[mode] = "cutlass.Int32"
        wrapper["capture_pdl_mode_param"] = mode
        # The semantic proof consumes the whole root. The selected body has
        # only the proved output and row global stores, and every other access is a
        # read (including the generated scalar prelude). Its existing alias
        # plan enumerates all of those argument storages for the runtime proof.
        wrapper["capture_pdl_contract"] = "paired-output-stores-v1"
        mode_arg = f", capture_pdl_wait_mode={mode}"
        if df.config.config.get("cute_pdl_release") == "loader_tail":
            mode_arg += ", capture_release_tail=True"
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
    if schedule.probability_schedule != MMA_DAG_PROBABILITY_SCHEDULE_SERIAL:
        assert schedule.paired_items and semantic.score_reference is not None
        panel_arg += f", probability_schedule={schedule.probability_schedule!r}"
    if semantic.score_reference is not None:
        assert semantic.denominator_per_key
        panel_arg += ", reference_scores=True"
    if semantic.score_base2:
        panel_arg += ", base2_scores=True"
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
    emitter.emit(
        f"{body_name}("
        f"{tensor_name(semantic.query)}, {tensor_name(semantic.page_table)}, "
        f"{tensor_name(semantic.output)}, {lse_arg}, "
        f"cutlass.Int32({item}), cutlass.Int64({begin}), cutlass.Int64({end}), "
        f"cutlass.Int32({length}), {coords.query_len}, {coords.group}, "
        f"{transport.query_rows}, {transport.heads}, {semantic.mask.window_left!r}, "
        f"cutlass.Float32({semantic.score_scale!r}), {schedule.fast_math!r}, "
        f"{', '.join(maps)}, coordinate_dtype=cutlass.{coordinate_dtype}{panel_arg}{io_arg}{query_arg}{mode_arg})"
    )
    for statement in ast.parse("\n".join(emitter.lines)).body:
        cg.add_statement(statement)
    wrappers = cute_codegen_state(cg).wrapper_plans
    wrappers.append(wrapper)
    alias_plan = mma_dag_alias_plan(df, selected.program)
    if guarded_pdl:
        loaded = {
            tensor_name(access)
            for access in selected.program.accesses
            if _is_call(access.node, memory_ops.load)
        }
        alias_plan["capture_loaded"] = [
            name in loaded for name in cast("list[str]", alias_plan["tensor_names"])
        ]
    wrappers.append(alias_plan)
    df.cute_state.lanes.owned_root_block_dims = (schedule.launch_warps * 32, 1, 1)
    df.cute_state.tcgen05.min_blocks_per_mp = schedule.min_blocks_per_mp
