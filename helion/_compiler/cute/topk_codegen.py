"""Emit register selection for a proven row-wise top-k region."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ...runtime.cute import topk as runtime_topk
from ..program_id import XYZProgramIDs
from ..tile_strategy import DeviceGridState
from .fx_matcher import _GeneratedCodeTemplate

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .topk import CuteTopKPlan


def codegen_topk_root(cg: GenerateAST, plan: CuteTopKPlan) -> bool:
    """Replace the matched root with a subgroup-local selection network.

    A signed rank for the 16-bit value and a reversed column index fit in
    31 bits for rows of at most 32768 elements. Int32 minimum padding cannot
    displace a real value, including infinity or NaN.
    Values are gathered from the original input after selection, preserving
    their exact bits independently of zero/NaN canonicalization in the keys.
    """
    fn = cg.device_function
    root = cg.current_root_graph_info
    if root is None or root.graph is not plan.root_graph or fn.pid is None:
        return False
    grid = cg.current_grid_state
    if not isinstance(grid, DeviceGridState):
        return False
    row_info = grid.block_id_to_info[plan.row_block_id]
    if row_info.begin_expr != 0 or row_info.end_expr != plan.x.size(0):
        return False
    pid_info = fn.pid.pid_info
    if len(pid_info) != 1 or pid_info[0].block_id != plan.row_block_id:
        return False
    fn.pid = XYZProgramIDs(
        pid_info=[pid_info[0]._replace(block_size_var=str(plan.rows_per_block))]
    )

    x_name = fn.tensor_arg(plan.x).name
    values_name = fn.tensor_arg(plan.values).name
    indices_name = fn.tensor_arg(plan.indices).name
    template = _GeneratedCodeTemplate(
        "topk",
        (*tuple(argument.name for argument in fn.arguments), *cg._extra_params),
        fn.new_var,
    )
    x = template.protect(x_name)
    values = template.protect(values_name)
    indices = template.protect(indices_name)
    # The persistent CuTe cache keys generated source rather than imported
    # helper code. Include the helper content in the actual imported symbol.
    helper_hash = hashlib.sha256(
        inspect.getsource(runtime_topk).encode("utf-8")
    ).hexdigest()[:16]
    helper_name = f"_cute_local_topk_{helper_hash}"
    cg.module_statements.append(
        ast.ImportFrom(
            module="helion.runtime.cute.topk",
            names=[ast.alias(name="local_topk", asname=helper_name)],
            level=0,
        )
    )
    index_bits = (plan.n - 1).bit_length()
    index_mask = (1 << index_bits) - 1
    padded_k = 1 << (plan.k - 1).bit_length()
    per_lane = (plan.n + plan.lanes_per_row - 1) // plan.lanes_per_row
    fragment_size = max(1 << (per_lane - 1).bit_length(), padded_k, plan.vector_width)
    infinity_bits = 0x7F80 if plan.x.dtype == torch.bfloat16 else 0x7C00
    reverse = "topk_ordered = -topk_ordered" if not plan.largest else ""
    encode = f"""
topk_magnitude = topk_bits & cutlass.Int32(32767)
topk_sign = topk_bits >> cutlass.Int32(31)
topk_ordered = (topk_magnitude ^ topk_sign) - topk_sign
if topk_magnitude > cutlass.Int32({infinity_bits}):
    topk_ordered = cutlass.Int32(32767)
{reverse}
topk_keys[topk_i] = ((topk_ordered << cutlass.Int32({index_bits}))
                    | (cutlass.Int32({index_mask}) - topk_col))
"""
    # Vector loads require an aligned logical base and complete aligned rows.
    # Irregular widths and sliced bases retain the scalar masked path.
    row_guard = (
        "True"
        if plan.x.size(0) % plan.rows_per_block == 0
        else f"topk_row < {x}.shape[0]"
    )
    group_guard = (
        "True"
        if plan.lanes_per_row * plan.rows_per_block >= 32
        else f"topk_group < cutlass.Int32({plan.rows_per_block})"
    )
    input_full = fragment_size * plan.lanes_per_row == plan.n
    scalar_col_guard = "True" if input_full else f"topk_col < cutlass.Int32({plan.n})"
    vector_col_guard = (
        "True" if input_full else f"topk_col_base < cutlass.Int32({plan.n})"
    )
    output_col_guard = (
        "True"
        if plan.k % plan.lanes_per_row == 0
        else f"topk_output_col < cutlass.Int32({plan.k})"
    )
    output_index_dtype = (
        "cutlass.Int32" if plan.indices.dtype == torch.int32 else "cutlass.Int64"
    )
    row_stride = plan.x.stride(0)
    max_input_offset = (
        max(row_stride, (plan.x.size(0) - 1) * row_stride + plan.n - 1)
        if isinstance(row_stride, int)
        else None
    )
    max_output_offset = plan.x.size(0) * plan.k - 1
    index_type = (
        "cutlass.Int32"
        if max_input_offset is not None
        and max(max_input_offset, max_output_offset) <= 2147483647
        else "cutlass.Int64"
    )
    vectorized = (
        plan.vector_width > 1
        and plan.n % plan.vector_width == 0
        and isinstance(row_stride, int)
        and row_stride % plan.vector_width == 0
    )
    vector_loads = f"""
for topk_chunk in cutlass.range({fragment_size // plan.vector_width}, unroll_full=True):
    topk_col_base = ((cutlass.Int32(topk_chunk * {plan.lanes_per_row})
                      + topk_lane) * cutlass.Int32({plan.vector_width}))
    if topk_valid_row & ({vector_col_guard}):
        topk_vector = cute.arch.load(
            topk_input_bits.iterator + topk_row * {index_type}({row_stride}) + {index_type}(topk_col_base),
            ir.VectorType.get([{plan.vector_width}], cutlass.Uint16.mlir_type),
        )
        for topk_element in cutlass.range_constexpr({plan.vector_width}):
            topk_i = topk_chunk * {plan.vector_width} + topk_element
            topk_col = topk_col_base + cutlass.Int32(topk_element)
            topk_bits = cutlass.Int32(cutlass.Uint16(topk_vector[topk_element]).bitcast(cutlass.Int16))
{textwrap.indent(encode, "            ")}
"""
    scalar_loads = f"""
for topk_i in cutlass.range({fragment_size}, unroll_full=True):
    topk_col = (
        (cutlass.Int32(topk_i // {plan.vector_width})
         * cutlass.Int32({plan.lanes_per_row}) + topk_lane)
        * cutlass.Int32({plan.vector_width})
        + cutlass.Int32(topk_i % {plan.vector_width})
    )
    if topk_valid_row & ({scalar_col_guard}):
        topk_bits = cutlass.Int32(topk_input_bits[topk_row, topk_col].bitcast(cutlass.Int16))
{textwrap.indent(encode, "        ")}
"""
    loads = scalar_loads
    if vectorized:
        loads = (
            f"if cutlass.const_expr(topk_input_bits.iterator.alignment >= {plan.vector_width * 2}):\n"
            + textwrap.indent(vector_loads, "    ")
            + "else:\n"
            + textwrap.indent(scalar_loads, "    ")
        )
    body = f"""
topk_thread = cutlass.Int32(cute.arch.thread_idx()[0])
topk_lane = topk_thread % cutlass.Int32({plan.lanes_per_row})
topk_group = topk_thread // cutlass.Int32({plan.lanes_per_row})
topk_row = ({index_type}(cute.arch.block_idx()[0])
            * {index_type}({plan.rows_per_block}) + {index_type}(topk_group))
topk_valid_row = ({row_guard}) & ({group_guard})
topk_input_bits = cute.make_tensor(
    cute.recast_ptr({x}.iterator, dtype=cutlass.Uint16), {x}.layout
)
topk_keys = cute.make_rmem_tensor({fragment_size}, cutlass.Int32)
topk_keys.fill(cutlass.Int32(-2147483648))
{loads}

topk_selected = {helper_name}(topk_keys, {padded_k}, {plan.lanes_per_row})
# Redistribute registers before storing so each memory instruction serves
# the entire lane subgroup instead of issuing one instruction per output rank.
topk_local = cute.make_rmem_tensor({(plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row}, cutlass.Int32)
topk_local.fill(cutlass.Int32(0))
for topk_output in cutlass.range_constexpr({plan.k}):
    if topk_lane == cutlass.Int32(topk_output % {plan.lanes_per_row}):
        topk_local[topk_output // {plan.lanes_per_row}] = topk_selected[topk_output]
for topk_j in cutlass.range_constexpr({(plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row}):
    topk_output_col = cutlass.Int32(topk_j * {plan.lanes_per_row}) + topk_lane
    if topk_valid_row & ({output_col_guard}):
        topk_selected_index = (
            cutlass.Int32({index_mask})
            - (topk_local[topk_j] & cutlass.Int32({index_mask}))
        )
        {values}[topk_row, topk_output_col] = {x}[topk_row, topk_selected_index]
        {indices}[topk_row, topk_output_col] = {output_index_dtype}(topk_selected_index)

"""
    statements: list[ast.AST] = []
    with cg.set_statements(statements):
        for statement in ast.parse(template.render(body)).body:
            cg.add_statement(statement)
    fn.preamble = []
    fn.body = statements
    fn.placeholder_args.update((x_name, values_name, indices_name))
    return True
