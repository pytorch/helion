"""CuTe codegen for tile reshape/permute operations.

Each thread holds one element of a tile. Shape operations (permute, reshape,
view) change which element each thread should hold. When the thread-to-element
mapping changes, data must be shuffled between threads via shared memory.

Key design: Per-dimension thread coordinates are determined by looking up
each tile dimension's block_id and finding its thread axis via
active_device_loops, NOT from the global thread block dimensions.
"""

from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx.node import Node
from torch.fx.node import map_arg

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .indexing import CuteShapeChainView
from .indexing import is_cute_shape_chain_target

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext
    from ..compile_environment import Config
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


def _env_arg(ctx: LoweringContext, node: Node) -> object:
    return ctx.env[node]


def _shape_chain_only_users(node: Node) -> bool:
    if not node.users:
        return False
    return all(
        user.op == "call_function" and is_cute_shape_chain_target(user.target)
        for user in node.users
    )


def _get_tile_shape(
    fake_tensor: torch.Tensor,
    env: CompileEnvironment,
    config: Config,
) -> list[int]:
    """Map a FakeTensor's symbolic dimensions to concrete tile (block) sizes."""
    shape: list[int] = []
    for dim_size in fake_tensor.shape:
        block_id = env.get_block_id(dim_size)
        if block_id is not None:
            # pyrefly: ignore [bad-argument-type]
            bs = env.block_sizes[block_id].from_config(config)
            if isinstance(bs, int):
                shape.append(int(bs))
                continue
        raw_expr = getattr(getattr(dim_size, "node", None), "_expr", None)
        if isinstance(raw_expr, sympy.Expr):
            replacements: dict[sympy.Symbol, sympy.Integer] = {}
            for symbol in raw_expr.free_symbols:
                if not isinstance(symbol, sympy.Symbol):
                    break
                block_id = env.get_block_id(symbol)
                if block_id is None:
                    break
                # pyrefly: ignore [bad-argument-type]
                bs = env.block_sizes[env.canonical_block_id(block_id)].from_config(
                    config
                )
                if not isinstance(bs, int):
                    break
                replacements[symbol] = sympy.Integer(bs)
            else:
                shape.append(int(raw_expr.xreplace(replacements)))
                continue
        with contextlib.suppress(Exception):
            shape.append(int(dim_size))
            continue
        shape.append(env.size_hint(dim_size))
    return shape


def _get_dim_local_coord(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
) -> str:
    """Get the current local coordinate expression for a tile dimension.

    Uses the current block-local index when the dimension is active, which
    preserves lane-loop coordinates as well as CUDA thread coordinates.
    """
    env = CompileEnvironment.current()
    dim_size = fake_tensor.shape[dim]
    block_id = env.get_block_id(dim_size)

    if block_id is None:
        return "cutlass.Int32(0)"

    if cg.active_device_loops.get(block_id):
        index_var = cg.index_var(block_id)
        offset_var = cg.offset_var(block_id)
        return f"(({index_var}) - ({offset_var}))"

    thread_axis = _get_dim_thread_axis(cg, fake_tensor, dim)
    if thread_axis is not None:
        return _grid_local_coord_expr(cg, block_id, thread_axis)

    return "cutlass.Int32(0)"


def _grid_local_coord_expr(
    cg: GenerateAST,
    block_id: int,
    thread_axis: int,
) -> str:
    """Return the current grid-local coordinate, including lane-loop offsets."""
    coord = f"cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])"
    if cg.current_grid_state is None:
        return coord

    strategy = cg.current_grid_state.strategy
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not isinstance(lane_vars, dict) or block_id not in lane_vars:
        return coord

    elements_per_thread_fn = getattr(strategy, "_elements_per_thread_for_block", None)
    if not callable(elements_per_thread_fn):
        return coord

    elements_per_thread = elements_per_thread_fn(block_id)
    lane_var = lane_vars[block_id]
    if elements_per_thread == 1:
        return f"{coord} + cutlass.Int32({lane_var})"
    return f"{coord} * cutlass.Int32({elements_per_thread}) + cutlass.Int32({lane_var})"


def _get_dim_thread_axis(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
) -> int | None:
    """Return the thread axis for a tile dimension, if one exists."""
    env = CompileEnvironment.current()
    dim_size = fake_tensor.shape[dim]
    block_id = env.get_block_id(dim_size)

    if block_id is None:
        return None

    loops = cg.active_device_loops.get(block_id)
    if loops:
        state = loops[-1]
        thread_axis = state.block_thread_axes.get(block_id)
        if thread_axis is not None:
            return thread_axis

    if cg.current_grid_state is not None:
        thread_axis = cg.current_grid_state.block_thread_axes.get(block_id)
        if thread_axis is not None:
            return thread_axis

    return None


def _dim_has_active_local_coord(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
) -> bool:
    env = CompileEnvironment.current()
    block_id = env.get_block_id(fake_tensor.shape[dim])
    if block_id is None:
        return False
    return bool(cg.active_device_loops.get(block_id))


def _permute_reorders_active_dims(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    perm: list[int],
) -> bool:
    active_dims = [
        dim
        for dim in range(len(perm))
        if _dim_has_active_local_coord(cg, fake_tensor, dim)
    ]
    return [dim for dim in perm if dim in active_dims] != active_dims


def _shape_op_needs_materialization(node: Node) -> bool:
    """Return True when non-store consumers need values, not just metadata."""
    from ...language import memory_ops

    matmul_targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    reduction_names = ("sum", "amax", "amin", "prod", "mean")
    for user in node.users:
        if user.op != "call_function":
            return True
        if user.target is memory_ops.store:
            continue
        # CuTe matmul fallbacks consume one scalar per thread/lane. In that mode a
        # transpose-like shape op only changes the logical operand layout, not the
        # scalar value held by the current thread.
        if user.target in matmul_targets:
            continue
        target_name = str(user.target)
        if any(name in target_name for name in reduction_names):
            continue
        return True
    return False


def _permute_needs_materialization(node: Node) -> bool:
    return _shape_op_needs_materialization(node)


def _flat_index_from_coords(
    coords: list[str],
    shape: list[int],
) -> str:
    """Convert ND coordinate expressions to a flat row-major index."""
    ndim = len(shape)
    if ndim == 1:
        return coords[0]
    parts: list[str] = []
    for i in range(ndim):
        stride = 1
        for j in range(i + 1, ndim):
            stride *= shape[j]
        if stride == 1:
            parts.append(f"({coords[i]})")
        else:
            parts.append(f"({coords[i]}) * cutlass.Int32({stride})")
    return " + ".join(parts)


def _coords_from_flat_index(
    flat_index: str,
    shape: list[int],
) -> list[str]:
    """Convert a row-major flat index to ND coordinates."""
    coords: list[str] = []
    ndim = len(shape)
    for i in range(ndim):
        size = shape[i]
        if size == 1:
            coords.append("cutlass.Int32(0)")
            continue
        stride = 1
        for j in range(i + 1, ndim):
            stride *= shape[j]
        if stride == 1:
            coords.append(f"({flat_index}) % cutlass.Int32({size})")
        else:
            coords.append(
                f"(({flat_index}) // cutlass.Int32({stride})) % cutlass.Int32({size})"
            )
    return coords


def _emit_cute_permute_shuffle(
    cg: GenerateAST,
    tensor: ast.AST,
    input_val: torch.Tensor,
    output_val: torch.Tensor,
    perm: list[int],
) -> ast.AST:
    env = CompileEnvironment.current()
    df = cg.device_function
    config = df.config

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)

    input_numel = 1
    for size in input_shape:
        input_numel *= size

    if input_numel == 1:
        return tensor

    dtype_str = env.backend.dtype_str(input_val.dtype)
    smem_ptr = df.new_var("permute_smem_ptr")
    smem = df.new_var("permute_smem")
    input_name = df.new_var("permute_input")
    result = df.new_var("permuted")

    src_coords = [
        _get_dim_local_coord(cg, input_val, i) for i in range(len(input_shape))
    ]
    current_flat = _flat_index_from_coords(src_coords, input_shape)

    # Preserve the current positional thread assignment, reinterpret that
    # position in the permuted output shape, then map back to the source
    # coordinates to fetch the transposed value.
    output_coords = _coords_from_flat_index(current_flat, output_shape)
    read_coords = [output_coords[perm.index(i)] for i in range(len(perm))]
    read_flat = _flat_index_from_coords(read_coords, input_shape)

    cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )
    cg.add_statement(statement_from_string(f"{input_name} = {{_inp}}", _inp=tensor))
    cg.add_statement(statement_from_string(f"{smem}[{current_flat}] = {input_name}"))
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    cg.add_statement(statement_from_string(f"{result} = {smem}[{read_flat}]"))
    return expr_from_string(result)


def _inverse_permute_coords(coords: list[str], perm: list[int]) -> list[str]:
    return [coords[perm.index(i)] for i in range(len(perm))]


def _expand_source_coords(
    output_coords: list[str],
    *,
    source_shape: list[int],
    output_shape: list[int],
) -> list[str] | None:
    rank_delta = len(output_shape) - len(source_shape)
    if rank_delta < 0:
        return None
    source_coords: list[str] = []
    for i, source_extent in enumerate(source_shape):
        output_dim = i + rank_delta
        output_extent = output_shape[output_dim]
        if source_extent == 1:
            source_coords.append("cutlass.Int32(0)")
        elif source_extent == output_extent:
            source_coords.append(output_coords[output_dim])
        else:
            return None
    return source_coords


def _stack_choice_expr(
    inputs: list[ast.AST],
    *,
    selector: str,
) -> ast.AST:
    selected = inputs[-1]
    for i in range(len(inputs) - 2, -1, -1):
        selected = expr_from_string(
            "({then_expr}) if ({selector}) == cutlass.Int32({i}) else ({else_expr})",
            then_expr=inputs[i],
            selector=expr_from_string(selector),
            i=ast.Constant(value=i),
            else_expr=selected,
        )
    return selected


def _resolve_shape_chain_expr(
    ctx: LoweringContext,
    node: Node,
    flat_index: str,
) -> ast.AST | None:
    env = CompileEnvironment.current()
    df = ctx.cg.device_function
    config = df.config
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        resolved = ctx.env.get(node)
        return resolved if isinstance(resolved, ast.AST) else None

    if node.target in (
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    ):
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        return _resolve_shape_chain_expr(ctx, source, flat_index)

    if node.target is torch.ops.aten.permute.default:
        source = node.args[0]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(source, Node) or not isinstance(dims, (list, tuple)):
            return None
        perm = [dim for dim in dims if isinstance(dim, int)]
        if len(perm) != len(dims):
            return None
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        source_coords = _inverse_permute_coords(coords, perm)
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.expand.default:
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        output_shape = _get_tile_shape(value, env, config)
        source_shape = _get_tile_shape(source_val, env, config)
        output_coords = _coords_from_flat_index(flat_index, output_shape)
        source_coords = _expand_source_coords(
            output_coords,
            source_shape=source_shape,
            output_shape=output_shape,
        )
        if source_coords is None:
            return None
        source_flat = _flat_index_from_coords(source_coords, source_shape)
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.transpose.int:
        source = node.args[0]
        dim0 = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1")
        if (
            not isinstance(source, Node)
            or not isinstance(dim0, int)
            or not isinstance(dim1, int)
        ):
            return None
        ndim = len(value.shape)
        dim0 %= ndim
        dim1 %= ndim
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        source_coords = _inverse_permute_coords(coords, perm)
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.t.default:
        source = node.args[0]
        if not isinstance(source, Node) or len(value.shape) != 2:
            return None
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        source_coords = [coords[1], coords[0]]
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.unsqueeze.default:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            return None
        ndim = len(value.shape)
        dim %= ndim
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        source_coords = coords[:dim] + coords[dim + 1 :]
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.squeeze.dim:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            return None
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        dim %= source_val.ndim
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        if int(source_val.shape[dim]) != 1:
            source_coords = coords
        else:
            source_coords = [*coords[:dim], "cutlass.Int32(0)", *coords[dim:]]
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat)

    if node.target is torch.ops.aten.stack.default:
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (list, tuple)) or not isinstance(dim, int):
            return None
        if not all(isinstance(tensor, Node) for tensor in tensors):
            return None
        dim %= len(value.shape)
        coords = _coords_from_flat_index(
            flat_index, _get_tile_shape(value, env, config)
        )
        selector = coords[dim]
        input_exprs: list[ast.AST] = []
        for tensor in tensors:
            assert isinstance(tensor, Node)
            input_val = tensor.meta.get("val")
            if not isinstance(input_val, torch.Tensor):
                return None
            input_coords = coords[:dim] + coords[dim + 1 :]
            input_flat = _flat_index_from_coords(
                input_coords,
                _get_tile_shape(input_val, env, config),
            )
            input_expr = _resolve_shape_chain_expr(ctx, tensor, input_flat)
            if input_expr is None:
                return None
            input_exprs.append(input_expr)
        return _stack_choice_expr(
            input_exprs,
            selector=selector,
        )

    resolved = ctx.env.get(node)
    if isinstance(resolved, CuteShapeChainView):
        return _resolve_shape_chain_expr(ctx, resolved.node, flat_index)
    return resolved if isinstance(resolved, ast.AST) else None


def _current_flat_index_for_value(ctx: LoweringContext, value: torch.Tensor) -> str:
    cg = cast("GenerateAST", ctx.cg)
    env = CompileEnvironment.current()
    config = cg.device_function.config
    shape = _get_tile_shape(value, env, config)
    coords = [_get_dim_local_coord(cg, value, i) for i in range(len(shape))]
    return _flat_index_from_coords(coords, shape)


def resolve_cute_shape_chain_value(
    ctx: LoweringContext,
    node: Node,
) -> ast.AST | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        resolved = ctx.env.get(node)
        return resolved if isinstance(resolved, ast.AST) else None
    return _resolve_shape_chain_expr(
        ctx, node, _current_flat_index_for_value(ctx, value)
    )


def codegen_cute_reshape(ctx: LoweringContext, node: Node) -> object:
    """Codegen for view/reshape on CuTe tiles."""
    from ..generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    # pyrefly: ignore [bad-argument-type]
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    shape_chain = tensor if isinstance(tensor, CuteShapeChainView) else None

    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    output_val = node.meta["val"]
    assert isinstance(input_val, torch.Tensor) and isinstance(output_val, torch.Tensor)

    env = CompileEnvironment.current()
    df = ctx.cg.device_function
    config = df.config

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)

    if input_shape == output_shape and isinstance(tensor, ast.AST):
        assert isinstance(tensor, ast.AST)
        return tensor

    source_node = shape_chain.node if shape_chain is not None else node.args[0]
    if shape_chain is not None and _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    if isinstance(source_node, Node):
        output_coords = [
            _get_dim_local_coord(ctx.cg, output_val, i)
            for i in range(len(output_shape))
        ]
        output_flat = _flat_index_from_coords(output_coords, output_shape)
        fused_expr = _resolve_shape_chain_expr(ctx, source_node, output_flat)
        if fused_expr is not None:
            return fused_expr

    if (
        ctx.cg.current_grid_state is not None
        and ctx.cg.current_grid_state.has_lane_loops()
        and not _shape_op_needs_materialization(node)
        and isinstance(tensor, ast.AST)
    ):
        assert isinstance(tensor, ast.AST)
        return tensor

    # Adding/removing unit dimensions is a no-op
    input_non_unit = [s for s in input_shape if s != 1]
    output_non_unit = [s for s in output_shape if s != 1]
    if input_non_unit == output_non_unit and isinstance(tensor, ast.AST):
        assert isinstance(tensor, ast.AST)
        return tensor

    input_numel = 1
    for s in input_shape:
        input_numel *= s

    if input_numel == 1:
        assert isinstance(tensor, ast.AST)
        return tensor

    dtype_str = env.backend.dtype_str(input_val.dtype)

    smem_ptr = df.new_var("reshape_smem_ptr")
    smem = df.new_var("reshape_smem")
    input_name = df.new_var("reshape_input")
    result = df.new_var("reshaped")

    # Get per-dimension thread coordinates using block_id → thread axis mapping
    src_coords = [
        _get_dim_local_coord(ctx.cg, input_val, i) for i in range(len(input_shape))
    ]
    src_flat = _flat_index_from_coords(src_coords, input_shape)
    output_coords = [
        _get_dim_local_coord(ctx.cg, output_val, i) for i in range(len(output_shape))
    ]
    output_flat = _flat_index_from_coords(output_coords, output_shape)

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

    if shape_chain is not None:
        tensor = _resolve_shape_chain_expr(ctx, shape_chain.node, src_flat)
    if not isinstance(tensor, ast.AST):
        raise TypeError(f"Expected AST for CuTe reshape input, got {type(tensor)}")
    ctx.cg.add_statement(statement_from_string(f"{input_name} = {{_inp}}", _inp=tensor))
    ctx.cg.add_statement(statement_from_string(f"{smem}[{src_flat}] = {input_name}"))
    ctx.cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    ctx.cg.add_statement(statement_from_string(f"{result} = {smem}[{output_flat}]"))

    return expr_from_string(result)


def codegen_cute_permute(ctx: LoweringContext, node: Node) -> object:
    """Codegen for permute/transpose on CuTe tiles.

    Uses shared memory to shuffle elements between threads. Each dimension's
    thread coordinate is determined by looking up its block_id's thread axis.
    """
    from ..generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    # pyrefly: ignore [bad-argument-type]
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    shape_chain = tensor if isinstance(tensor, CuteShapeChainView) else None

    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    assert isinstance(input_val, torch.Tensor)

    # pyrefly: ignore [not-iterable]
    perm = [*dims]
    ndim = len(input_val.shape)
    assert len(perm) == ndim
    needs_materialization = _permute_reorders_active_dims(
        ctx.cg, input_val, perm
    ) and _permute_needs_materialization(node)

    if perm == list(range(ndim)) and isinstance(tensor, ast.AST):
        assert isinstance(tensor, ast.AST)
        return tensor

    source_node = shape_chain.node if shape_chain is not None else node.args[0]
    output_val = node.meta["val"]
    if (
        shape_chain is not None
        and not needs_materialization
        and _shape_chain_only_users(node)
    ):
        return CuteShapeChainView(node)
    if (
        not needs_materialization
        and isinstance(source_node, Node)
        and isinstance(output_val, torch.Tensor)
    ):
        output_shape = _get_tile_shape(
            output_val, CompileEnvironment.current(), ctx.cg.device_function.config
        )
        output_coords = [
            _get_dim_local_coord(ctx.cg, output_val, i)
            for i in range(len(output_shape))
        ]
        output_flat = _flat_index_from_coords(output_coords, output_shape)
        fused_expr = _resolve_shape_chain_expr(ctx, source_node, output_flat)
        if fused_expr is not None:
            return fused_expr

    if not needs_materialization:
        if isinstance(tensor, ast.AST):
            return tensor
        raise TypeError(f"Expected AST for CuTe permute input, got {type(tensor)}")

    assert isinstance(output_val, torch.Tensor)
    if shape_chain is not None:
        input_shape = _get_tile_shape(
            input_val, CompileEnvironment.current(), ctx.cg.device_function.config
        )
        src_coords = [
            _get_dim_local_coord(ctx.cg, input_val, i) for i in range(len(input_shape))
        ]
        src_flat = _flat_index_from_coords(src_coords, input_shape)
        tensor = _resolve_shape_chain_expr(ctx, shape_chain.node, src_flat)
    if not isinstance(tensor, ast.AST):
        raise TypeError(f"Expected AST for CuTe permute input, got {type(tensor)}")
    return _emit_cute_permute_shuffle(ctx.cg, tensor, input_val, output_val, perm)


def codegen_cute_store_permute(
    state: CodegenState,
    tensor: ast.AST,
    permute_node: Node,
) -> ast.AST | None:
    """Materialize a permute when a store needs the transposed values."""
    from ..generate_ast import GenerateAST

    if not isinstance(state.codegen, GenerateAST):
        return None

    if _permute_needs_materialization(permute_node):
        return None

    info = _store_permute_info(permute_node)
    if info is None:
        return None
    input_node, perm = info

    input_val = input_node.meta.get("val")
    output_val = permute_node.meta.get("val")
    if not isinstance(input_val, torch.Tensor) or not isinstance(
        output_val, torch.Tensor
    ):
        return None

    if not _permute_reorders_active_dims(state.codegen, input_val, perm):
        return tensor

    return _emit_cute_permute_shuffle(
        state.codegen,
        tensor,
        input_val,
        output_val,
        perm,
    )


def _store_permute_info(node: Node) -> tuple[Node, list[int]] | None:
    """Normalize transpose-like store inputs to an explicit permutation."""
    if node.op != "call_function" or not node.args:
        return None

    input_node = node.args[0]
    if not isinstance(input_node, Node):
        return None

    input_val = input_node.meta.get("val")
    if not isinstance(input_val, torch.Tensor):
        return None

    target = node.target
    if target is torch.ops.aten.permute.default:
        if len(node.args) < 2:
            return None
        dims = node.args[1]
        if not isinstance(dims, (list, tuple)):
            return None
        perm: list[int] = []
        for dim in dims:
            if not isinstance(dim, int):
                return None
            perm.append(dim)
        return input_node, perm

    ndim = input_val.ndim
    if target is torch.ops.aten.transpose.int:
        if len(node.args) < 3:
            return None
        dim0_arg = node.args[1]
        dim1_arg = node.args[2]
        if not isinstance(dim0_arg, int) or not isinstance(dim1_arg, int):
            return None
        dim0 = dim0_arg % ndim
        dim1 = dim1_arg % ndim
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return input_node, perm

    if target is torch.ops.aten.t.default:
        if ndim != 2:
            return None
        return input_node, [1, 0]

    return None
