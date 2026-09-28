from __future__ import annotations

import ast
import hashlib
import inspect
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_reshape import _coords_from_flat_index
from .cute_reshape import _current_flat_index_for_value
from .cute_reshape import _flat_index_from_coords
from .cute_reshape import _get_dim_local_coord
from .cute_reshape import _get_tile_shape
from .cute_reshape import resolve_cute_shape_chain_value
from .indexing import CuteShapeChainView

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import LoopedReductionStrategy
    from ..reduction_strategy import PersistentReductionStrategy
    from ..tile_strategy import DeviceLoopState


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
    env = CompileEnvironment.current()
    terms: list[str] = []
    for dim, size in enumerate(value.shape):
        if _current_global_coord_expr(cg, value, dim) is None:
            continue
        if not isinstance(size, (int, torch.SymInt)):
            continue
        size_expr = (
            repr(size)
            if isinstance(size, int)
            else cg.device_function.sympy_expr(size._sympy_())
        )
        # Tensor dimensions describe this tile, not the whole input. A global
        # coordinate would reject every CTA after the first one. The existing
        # loop/grid mask separately carries the true input tail (and surplus
        # thread bounds), so padded zero loads never become valid candidates.
        coord = _get_dim_local_coord(cg, value, dim)
        terms.append(f"(({coord}) >= 0 and ({coord}) < ({size_expr}))")
        block_id = env.get_block_id(size)
        assert block_id is not None
        loops = cg.active_device_loops.get(block_id)
        if loops:
            mask = loops[-1].strategy.mask_var(block_id)
        else:
            assert cg.current_grid_state is not None
            mask = cg.current_grid_state.strategy.mask_var(block_id)
        if mask is not None:
            terms.append(mask)
    if not terms:
        return None
    return " and ".join(terms)


def _argreduce_source_expr(ctx: LoweringContext, input_node: Node) -> ast.AST:
    source = ctx.env[input_node]
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
    assert isinstance(lane_vars, dict)
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
        # ``lane_loops`` records the full elements-per-thread, but a vec'd
        # lane loop runs its outer var over extent // V with a constexpr-V
        # inner loop: the last element is the last vec lane of the last
        # outer iteration.
        wrapper = grid_state.vec_lane_wrappers.get(lane_var)
        vec_width = getattr(strategy, "_cute_lane_vec_width_by_block", {}).get(
            block_id, 1
        )
        if wrapper is not None and vec_width > 1 and extent % vec_width == 0:
            terms.append(
                f"(({lane_var}) == {extent // vec_width - 1} "
                f"and ({wrapper.vec_lane_var}) == {vec_width - 1})"
            )
        else:
            terms.append(f"(({lane_var}) == {extent - 1})")
    if not terms:
        return None
    return " and ".join(terms)


def _import_ordered_key_helper(cg: GenerateAST) -> str:
    from ...runtime.cute import ordered_key

    # CuTe's persistent cache keys generated source, including import aliases.
    digest = hashlib.sha256(inspect.getsource(ordered_key).encode()).hexdigest()[:16]
    encoder = f"_cute_argreduce_key_{digest}"
    cg.module_statements.append(
        ast.ImportFrom(
            module="helion.runtime.cute.ordered_key",
            names=[
                ast.alias(name="encode_ordered_key_16", asname=encoder),
            ],
            level=0,
        )
    )
    return encoder


def codegen_packed_reduction(
    state: CodegenState,
    strategy: LoopedReductionStrategy | PersistentReductionStrategy,
    device_loop: DeviceLoopState | None,
    input_name: str,
    reduction_type: str,
    dim: int,
    fake_input: torch.Tensor,
    fake_output: torch.Tensor,
) -> ast.AST | None:
    """Use one ordered-key collective for bounded low-precision argreductions."""
    from ..reduction_strategy import LoopedReductionStrategy

    env = CompileEnvironment.current()
    if isinstance(strategy, LoopedReductionStrategy):
        if (
            strategy._cute_rolled_cluster_n != 1
            or strategy._cute_emitted_vec_load
            or strategy._cute_reduction_vec_width != 1
        ):
            return None
    elif strategy._synthetic_cute_lane_var is not None:
        return None
    numel = env.block_sizes[strategy.block_index].numel
    threads = strategy._reduction_thread_count()
    if (
        fake_input.dtype not in (torch.float16, torch.bfloat16)
        or not isinstance(numel, (int, sympy.Integer))
        or not 0 < int(numel) <= 32768
        or not 0 < threads <= 32
        or threads & (threads - 1)
        or strategy._get_thread_axis() != 0
    ):
        return None
    extent = int(numel)
    index_bits = (extent - 1).bit_length()
    index_mask = (1 << index_bits) - 1
    infinity_bits = 0x7C00 if fake_input.dtype == torch.float16 else 0x7F80
    largest = reduction_type == "argmax"
    nan_order = "last" if largest else "first"
    fn = state.device_function
    encoder = _import_ordered_key_helper(state.codegen)
    assert state.fx_node is not None
    accumulator = fn.new_var(f"{state.fx_node.name}_key_acc", dce=True)
    result = fn.new_var(state.fx_node.name, dce=True)
    index = strategy.index_var(strategy.block_index)
    input_type = env.backend.dtype_str(fake_input.dtype)
    if device_loop is not None:
        local_value = fn.new_var(f"{state.fx_node.name}_local_value", dce=True)
        local_index = fn.new_var(f"{state.fx_node.name}_local_index", dce=True)
        value = fn.new_var(f"{state.fx_node.name}_value", dce=True)
        extreme = "float('-inf')" if largest else "float('inf')"
        device_loop.outer_prefix.extend(
            [
                statement_from_string(f"{local_value} = cutlass.Float32({extreme})"),
                statement_from_string(
                    f"{local_index} = cutlass.Int32(cute.arch.thread_idx()[0])"
                ),
            ]
        )
        state.add_statement(f"{value} = cutlass.Float32({input_type}({input_name}))")
        comparison = ">" if largest else "<"
        # Scalar reduction loops visit each lane's columns in increasing order,
        # starting at its thread index. Retaining equal values gives first-index
        # ties, including signed zero and infinities. The first NaN wins as well.
        better = (
            f"({index}) < {extent} and "
            f"(({value} {comparison} {local_value}) or "
            f"({value} != {value} and {local_value} == {local_value}))"
        )
        state.add_statement(
            f"{local_value}, {local_index} = ({value}, {index}) if ({better}) "
            f"else ({local_value}, {local_index})"
        )
        input_name, index = local_value, local_index
    encoded = (
        f"{encoder}({input_type}({input_name}).bitcast(cutlass.Uint16), "
        f"cutlass.Int32({index}), {index_bits}, {largest!r}, 'signed', "
        f"{infinity_bits}, nan_order={nan_order!r})"
    )
    # Invalid loads may use an extreme value also present in the input. Mask
    # keys explicitly so padding cannot win even when every real value is inf.
    encoded = f"({encoded} if ({index}) < {extent} else cutlass.Int32(-2147483648))"
    if device_loop is None:
        state.add_statement(f"{accumulator} = {encoded}")
    else:
        # Encode only each thread's winner: packing every loop element adds
        # work without reducing the number of subgroup collectives.
        device_loop.outer_suffix.append(
            statement_from_string(f"{accumulator} = {encoded}")
        )
    maximum = env.backend.reduction_expr(
        accumulator, "max", dim, threads_in_group=threads, dtype=torch.int32
    )
    decoded = (
        f"(cutlass.Int32({index_mask}) - ({maximum} & cutlass.Int32({index_mask})))"
    )
    decoded = strategy.maybe_reshape(decoded, dim, fake_input, fake_output)
    output_type = env.backend.dtype_str(fake_output.dtype)
    assignment = statement_from_string(f"{result} = {output_type}({decoded})")
    if device_loop is None:
        state.codegen.add_statement(assignment)
    else:
        device_loop.outer_suffix.append(assignment)
    return expr_from_string(result)


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
    start_expr = "cutlass.Int32(0)"
    stride_expr = "cutlass.Int32(1)"
    extent = input_numel

    if dim is not None:
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
        start_expr = initial_flat
        extent = input_shape[dim]
        if extent > 1:
            if keepdim:
                source_coords = [*output_coords]
                source_coords[dim] = "cutlass.Int32(1)"
            else:
                source_coords = [
                    *output_coords[:dim],
                    "cutlass.Int32(1)",
                    *output_coords[dim:],
                ]
            stride_expr = f"(({_flat_index_from_coords(source_coords, input_shape)}) - ({initial_flat}))"

    result_init = backend.cast_expr("0", output_dtype_str)
    ctx.cg.add_statement(statement_from_string(f"{result} = {result_init}"))
    scan_statements: list[ast.stmt] = [
        statement_from_string(
            f"{best_index} = _cute_argreduce_index("
            f"{smem}, {valid_smem}, {start_expr}, {stride_expr}, "
            f"extent={extent}, reduction_type={reduction_type!r})"
        ),
        statement_from_string(
            f"{result} = {backend.cast_expr(best_index, output_dtype_str)}"
        ),
    ]
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
