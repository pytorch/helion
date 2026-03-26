from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_reshape import _coords_from_flat_index
from .cute_reshape import _current_flat_index_for_value
from .cute_reshape import _flat_index_from_coords
from .cute_reshape import _get_tile_shape
from .cute_reshape import resolve_cute_shape_chain_value
from .indexing import CuteShapeChainView

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST


def _env_arg(ctx: LoweringContext, node: Node) -> object:
    return ctx.env[node]


def _argreduce_extreme_literal(
    dtype: torch.dtype,
    reduction_type: str,
) -> str:
    if dtype.is_floating_point:
        if reduction_type == "argmax":
            return "float('-inf')"
        return "float('inf')"
    if dtype == torch.bool:
        return "False" if reduction_type == "argmax" else "True"
    info = torch.iinfo(dtype)
    if reduction_type == "argmax":
        return repr(info.min)
    return repr(info.max)


def _current_global_coord_expr(
    cg: GenerateAST,
    value: torch.Tensor,
    dim: int,
) -> str | None:
    env = CompileEnvironment.current()
    block_id = env.get_block_id(value.shape[dim])
    if block_id is None:
        return None
    loops = cg.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.index_var(block_id)
    if (
        cg.current_grid_state is not None
        and block_id in cg.current_grid_state.block_ids
    ):
        return cg.current_grid_state.strategy.index_var(block_id)
    return None


def _input_in_bounds_expr(
    cg: GenerateAST,
    value: torch.Tensor,
) -> str | None:
    terms: list[str] = []
    for dim, size in enumerate(value.shape):
        coord = _current_global_coord_expr(cg, value, dim)
        if coord is None:
            continue
        if not isinstance(size, (int, torch.SymInt)):
            continue
        size_expr = (
            repr(size)
            if isinstance(size, int)
            else cg.device_function.sympy_expr(size._sympy_())
        )
        terms.append(f"(({coord}) < ({size_expr}))")
    if not terms:
        return None
    return " and ".join(terms)


def _argreduce_source_expr(ctx: LoweringContext, input_node: Node) -> ast.AST:
    source = _env_arg(ctx, input_node)
    if isinstance(source, CuteShapeChainView):
        resolved = resolve_cute_shape_chain_value(ctx, input_node)
        if resolved is None:
            raise TypeError("Expected resolvable CuTe shape chain value for argreduce")
        return resolved
    if not isinstance(source, ast.AST):
        raise TypeError(f"Expected AST for CuTe argreduce input, got {type(source)}")
    return source


def _cute_lane_loop_dims(cg: GenerateAST, value: torch.Tensor) -> set[int]:
    grid_state = cg.current_grid_state
    if grid_state is None or not grid_state.has_lane_loops():
        return set()
    strategy = grid_state.strategy
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not isinstance(lane_vars, dict) or not lane_vars:
        return set()
    env = CompileEnvironment.current()
    dims: set[int] = set()
    for dim, dim_size in enumerate(value.shape):
        block_id = env.get_block_id(dim_size)
        if block_id is not None and block_id in lane_vars:
            dims.add(dim)
    return dims


def _argreduce_scan_ready_expr(
    cg: GenerateAST,
    value: torch.Tensor,
    dim: int | None,
) -> str | None:
    lane_loop_dims = _cute_lane_loop_dims(cg, value)
    if not lane_loop_dims:
        return None
    reduced_dims = lane_loop_dims if dim is None else lane_loop_dims.intersection({dim})
    if not reduced_dims:
        return None

    grid_state = cg.current_grid_state
    assert grid_state is not None
    strategy = grid_state.strategy
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not isinstance(lane_vars, dict):
        return None
    lane_extents = dict(grid_state.lane_loops)
    env = CompileEnvironment.current()
    terms: list[str] = []
    for reduced_dim in sorted(reduced_dims):
        block_id = env.get_block_id(value.shape[reduced_dim])
        if block_id is None:
            continue
        lane_var = lane_vars.get(block_id)
        if lane_var is None:
            continue
        extent = lane_extents.get(lane_var)
        if extent is None or extent <= 0:
            continue
        terms.append(f"(({lane_var}) == {extent - 1})")
    if not terms:
        return None
    return " and ".join(terms)


def codegen_cute_tile_argreduce(
    ctx: LoweringContext,
    node: Node,
    reduction_type: str,
    *,
    dim: int | None,
    keepdim: bool,
) -> ast.AST:
    from ..generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    input_node = cast("Node", node.args[0])
    input_val = input_node.meta["val"]
    output_val = node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    assert isinstance(output_val, torch.Tensor)

    env = CompileEnvironment.current()
    backend = env.backend
    dtype_str = backend.dtype_str(input_val.dtype)
    valid_dtype_str = backend.dtype_str(torch.int32)
    output_dtype_str = backend.dtype_str(output_val.dtype)
    input_shape = _get_tile_shape(input_val, env, ctx.cg.device_function.config)
    output_shape = _get_tile_shape(output_val, env, ctx.cg.device_function.config)
    input_numel = math.prod(input_shape)
    input_flat = _current_flat_index_for_value(ctx, input_val)
    output_flat = _current_flat_index_for_value(ctx, output_val)
    source_expr = _argreduce_source_expr(ctx, input_node)
    invalid_literal = _argreduce_extreme_literal(input_val.dtype, reduction_type)
    invalid_value = backend.cast_expr(invalid_literal, dtype_str)
    in_bounds = _input_in_bounds_expr(ctx.cg, input_val)
    stored_value = source_expr
    stored_valid = backend.cast_expr("1", valid_dtype_str)
    if in_bounds is not None:
        stored_value = expr_from_string(
            f"({{value}} if {in_bounds} else {invalid_value})",
            value=source_expr,
        )
        stored_valid = f"({backend.cast_expr('1', valid_dtype_str)} if {in_bounds} else {backend.cast_expr('0', valid_dtype_str)})"

    smem_ptr = ctx.cg.device_function.new_var("argreduce_smem_ptr")
    smem = ctx.cg.device_function.new_var("argreduce_smem")
    valid_smem_ptr = ctx.cg.device_function.new_var("argreduce_valid_smem_ptr")
    valid_smem = ctx.cg.device_function.new_var("argreduce_valid_smem")
    best_value = ctx.cg.device_function.new_var(f"{reduction_type}_best_value")
    best_valid = ctx.cg.device_function.new_var(f"{reduction_type}_best_valid")
    best_index = ctx.cg.device_function.new_var(f"{reduction_type}_best_index")
    result = ctx.cg.device_function.new_var(reduction_type)

    ctx.cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    ctx.cg.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )
    ctx.cg.add_statement(
        statement_from_string(
            f"{valid_smem_ptr} = cute.arch.alloc_smem({valid_dtype_str}, {input_numel})"
        )
    )
    ctx.cg.add_statement(
        statement_from_string(
            f"{valid_smem} = cute.make_tensor({valid_smem_ptr}, ({input_numel},))"
        )
    )
    ctx.cg.add_statement(
        statement_from_string(f"{smem}[{input_flat}] = {{value}}", value=stored_value)
    )
    ctx.cg.add_statement(
        statement_from_string(f"{valid_smem}[{input_flat}] = {stored_valid}")
    )
    ctx.cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    scan_ready = _argreduce_scan_ready_expr(ctx.cg, input_val, dim)
    scan_statements: list[ast.stmt] = [
        statement_from_string(f"{best_index} = cutlass.Int64(0)")
    ]

    if dim is None:
        scan_statements.extend(
            [
                statement_from_string(f"{best_value} = {smem}[0]"),
                statement_from_string(f"{best_valid} = {valid_smem}[0]"),
            ]
        )
        for flat_idx in range(1, input_numel):
            candidate = ctx.cg.device_function.new_var(
                f"{reduction_type}_candidate_{flat_idx}"
            )
            candidate_valid = ctx.cg.device_function.new_var(
                f"{reduction_type}_candidate_valid_{flat_idx}"
            )
            compare = (
                f"(({candidate}) > ({best_value}))"
                f" or ((({candidate}) == ({best_value})) and (cutlass.Int64({flat_idx}) < ({best_index})))"
            )
            if reduction_type == "argmin":
                compare = (
                    f"(({candidate}) < ({best_value}))"
                    f" or ((({candidate}) == ({best_value})) and (cutlass.Int64({flat_idx}) < ({best_index})))"
                )
            better = (
                f"(({candidate_valid}) != cutlass.Int32(0) and ({best_valid}) == cutlass.Int32(0))"
                f" or ((({candidate_valid}) != cutlass.Int32(0)) and (({best_valid}) != cutlass.Int32(0)) and ({compare}))"
            )
            scan_statements.extend(
                [
                    statement_from_string(f"{candidate} = {smem}[{flat_idx}]"),
                    statement_from_string(
                        f"{candidate_valid} = {valid_smem}[{flat_idx}]"
                    ),
                    statement_from_string(
                        f"""if {better}:
    {best_value} = {candidate}
    {best_valid} = {candidate_valid}
    {best_index} = cutlass.Int64({flat_idx})"""
                    ),
                ]
            )
    else:
        output_coords = _coords_from_flat_index(output_flat, output_shape)
        if keepdim:
            initial_coords = [*output_coords]
            initial_coords[dim] = "cutlass.Int32(0)"
        else:
            initial_coords = [
                *output_coords[:dim],
                "cutlass.Int32(0)",
                *output_coords[dim:],
            ]
        initial_flat = _flat_index_from_coords(initial_coords, input_shape)
        scan_statements.extend(
            [
                statement_from_string(f"{best_value} = {smem}[{initial_flat}]"),
                statement_from_string(f"{best_valid} = {valid_smem}[{initial_flat}]"),
            ]
        )
        reduce_extent = input_shape[dim]
        for reduce_idx in range(1, reduce_extent):
            if keepdim:
                source_coords = [*output_coords]
                source_coords[dim] = f"cutlass.Int32({reduce_idx})"
            else:
                source_coords = [
                    *output_coords[:dim],
                    f"cutlass.Int32({reduce_idx})",
                    *output_coords[dim:],
                ]
            source_flat = _flat_index_from_coords(source_coords, input_shape)
            candidate = ctx.cg.device_function.new_var(
                f"{reduction_type}_candidate_{reduce_idx}"
            )
            candidate_valid = ctx.cg.device_function.new_var(
                f"{reduction_type}_candidate_valid_{reduce_idx}"
            )
            compare = (
                f"(({candidate}) > ({best_value}))"
                f" or ((({candidate}) == ({best_value})) and (cutlass.Int64({reduce_idx}) < ({best_index})))"
            )
            if reduction_type == "argmin":
                compare = (
                    f"(({candidate}) < ({best_value}))"
                    f" or ((({candidate}) == ({best_value})) and (cutlass.Int64({reduce_idx}) < ({best_index})))"
                )
            better = (
                f"(({candidate_valid}) != cutlass.Int32(0) and ({best_valid}) == cutlass.Int32(0))"
                f" or ((({candidate_valid}) != cutlass.Int32(0)) and (({best_valid}) != cutlass.Int32(0)) and ({compare}))"
            )
            scan_statements.extend(
                [
                    statement_from_string(f"{candidate} = {smem}[{source_flat}]"),
                    statement_from_string(
                        f"{candidate_valid} = {valid_smem}[{source_flat}]"
                    ),
                    statement_from_string(
                        f"""if {better}:
    {best_value} = {candidate}
    {best_valid} = {candidate_valid}
    {best_index} = cutlass.Int64({reduce_idx})"""
                    ),
                ]
            )

    result_init = backend.cast_expr("0", output_dtype_str)
    ctx.cg.add_statement(statement_from_string(f"{result} = {result_init}"))
    scan_statements.append(
        statement_from_string(
            f"{result} = {backend.cast_expr(best_index, output_dtype_str)}"
        )
    )
    if scan_ready is None:
        for stmt in scan_statements:
            ctx.cg.add_statement(stmt)
    else:
        ctx.cg.add_statement(
            ast.If(
                test=cast("ast.expr", expr_from_string(scan_ready)),
                body=scan_statements,
                orelse=[],
            )
        )
    return expr_from_string(result)
