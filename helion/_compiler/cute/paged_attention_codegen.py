"""Render the selected paged-attention schedule into an ordinary root."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from .cute_mma_dag_codegen import emit_mma_dag_module_statements
from .cute_mma_dag_codegen import mma_dag_alias_plan
from .cute_mma_dag_emitter_base import MmaDagEmitterBase
from .cute_mma_dag_emitter_base import _Evaluator
from .paged_attention_transport import paged_attention_launch_warps
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
    )
    emit_mma_dag_module_statements(cg)
    cg.ensure_module_statements(
        "import helion._compiler.cute._mma_dag_tcgen05_runtime as _helion_mma_dag_tc\n"
    )

    if semantic.denominator_per_key:
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
            (
                semantic.key.raw,
                semantic.key.scale,
                semantic.value.raw,
                semantic.value.scale,
            ),
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
    coords = semantic.coordinates
    coordinate_dtype = "Int64" if coords.key_dtype == torch.int64 else "Int32"
    panel_arg = (
        f", query_columns={schedule.query_columns}, scoped_registers={schedule.scoped_registers!r}"
        if semantic.denominator_per_key
        else ""
    )
    if semantic.score_reference is not None:
        assert semantic.denominator_per_key
        panel_arg += ", reference_scores=True"
        if semantic.score_base2:
            panel_arg += ", base2_scores=True"
    if coords.reflected is not None:
        assert semantic.denominator_per_key
        reflected_dtype = "Int64" if coords.reflected.dtype == torch.int64 else "Int32"
        panel_arg += f", reflected_dtype=cutlass.{reflected_dtype}"
    io_arg = ""
    if transport.direct_io is not None:
        assert semantic.denominator_per_key
        io = transport.direct_io
        io_arg = f", io_layouts={(io.query, io.output, io.lse)!r}"
    lse_arg = tensor_name(semantic.lse) if semantic.lse is not None else "None"
    emitter.emit(
        f"{body_name}("
        f"{tensor_name(semantic.query)}, {tensor_name(semantic.page_table)}, "
        f"{tensor_name(semantic.output)}, {lse_arg}, "
        f"cutlass.Int32({item}), cutlass.Int64({begin}), cutlass.Int64({end}), "
        f"cutlass.Int32({length}), {coords.query_len}, {coords.group}, "
        f"{transport.query_rows}, {transport.heads}, {semantic.mask.window_left!r}, "
        f"cutlass.Float32({semantic.score_scale!r}), {schedule.fast_math!r}, "
        f"{', '.join(maps)}, coordinate_dtype=cutlass.{coordinate_dtype}{panel_arg}{io_arg})"
    )
    for statement in ast.parse("\n".join(emitter.lines)).body:
        cg.add_statement(statement)
    wrappers = cute_codegen_state(cg).wrapper_plans
    wrappers.append(wrapper)
    wrappers.append(mma_dag_alias_plan(df, selected.program))
    df.cute_state.lanes.owned_root_block_dims = (schedule.launch_warps * 32, 1, 1)
    df.cute_state.tcgen05.min_blocks_per_mp = schedule.min_blocks_per_mp
