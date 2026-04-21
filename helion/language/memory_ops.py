from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _symint_expr
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.variable_origin import GridOrigin
from .._compiler.variable_origin import TileBeginOrigin
from .._compiler.variable_origin import TileCountOrigin
from .._compiler.variable_origin import TileEndOrigin
from .._compiler.variable_origin import TileIdOrigin
from . import _decorators
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.tile_strategy import LoopDimInfo

from .._compiler.host_function import SymbolOrigin

# TileBeginWithOffset removed - using TileBeginWithOffsetPattern instead

__all__ = ["load", "store"]

log = logging.getLogger(__name__)


# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor / stack tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | tuple,
    list[object],
    torch.Tensor | torch.SymInt | float | int,
    torch.Tensor | None,
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes_for_index(index)

    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, value, extra_mask)

    if isinstance(tensor, torch.Tensor):
        return (tensor, index, value, extra_mask)

    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None


@_decorators.codegen(store, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        fx_node = state.fx_node
        assert fx_node is not None
        epilogue_subtile_group_id = fx_node.meta.get("epilogue_subtile_group_id")
        if epilogue_subtile_group_id is None:
            indexing_idx = device_fn.allocate_store_index()
        elif fx_node.meta.get("epilogue_subtile_primary_store", False):
            indexing_idx = device_fn.allocate_store_index()
            device_fn.epilogue_subtile_store_indices[epilogue_subtile_group_id] = (
                indexing_idx
            )
        else:
            indexing_idx = device_fn.epilogue_subtile_store_indices[
                epilogue_subtile_group_id
            ]
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        if state.codegen.store_transform is not None:
            return state.codegen.store_transform(
                state,
                tensor,
                [*subscript],
                value,
                extra_mask,
                strategy.codegen_store,
            )

        return strategy.codegen_store(state, tensor, [*subscript], value, extra_mask)
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack stores (multi-tensor device pointers);
        # fall through to the unfused path regardless of store_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


def _can_tile_dimension(state: CodegenState, tensor_dim: int) -> bool:
    assert state.fx_node is not None
    tensor_arg_node = state.fx_node.args[0]  # 0th argument to load/store is the tensor
    assert isinstance(tensor_arg_node, torch.fx.Node)

    tensor_val = tensor_arg_node.meta.get("val")
    assert isinstance(tensor_val, torch.Tensor)

    dim_tilings = state.device_function.pallas_tensor_dim_tilings.get(id(tensor_val))
    assert isinstance(dim_tilings, list)
    assert tensor_dim < len(dim_tilings)
    from helion._compiler.pallas.plan_tiling import DimensionTiling

    assert isinstance(dim_tilings[tensor_dim], DimensionTiling)
    return dim_tilings[tensor_dim].can_tile


def _pallas_index_str(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    tensor: torch.Tensor,
) -> tuple[str, list[int]]:
    """Build a JAX/Pallas index string from a Helion subscript list.

    Uses ``pl.ds(offset, block_size)`` only for dimensions inside a looped
    reduction (``DeviceLoopState``).  Grid dimensions and persistent
    reduction dimensions use ``...`` — Pallas BlockSpecs in the launcher
    handle the grid-level tiling.

    For ``EmitPipelineLoopState`` or ``ForiLoopState``, pipeline-tiled
    dimensions also use ``...`` since the pipeline handles that tiling
    (via BlockSpecs or DMA copies respectively).

    Also returns positions of ``None`` indices so the caller can apply
    ``jnp.expand_dims`` after loading.
    """
    from .._compiler.tile_strategy import EmitPipelineLoopState
    from .._compiler.tile_strategy import ForiLoopState

    if not subscript:
        return "...", []

    # Check if we're inside an emit_pipeline or fori_loop with DMA.
    # When fori_loop runs without DMA (use_dma=False), its block_ids
    # are NOT treated as pipeline dims — they get pl.ds() slicing instead.
    in_pipeline = False
    pipeline_block_ids: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, EmitPipelineLoopState) or (
                isinstance(loop, ForiLoopState) and loop.use_dma
            ):
                in_pipeline = True
                pipeline_block_ids.update(loop.block_ids)

    # Use pre-computed indexing patterns from plan_tiling analysis
    indexing_patterns = _pallas_get_indexing_patterns(state, tensor)

    # Build parts using the pre-computed patterns
    parts: list[str] = []
    none_dims: list[int] = []
    out_pos = 0
    tensor_dim = 0

    for i, (idx, pattern) in enumerate(zip(subscript, indexing_patterns, strict=True)):
        if idx is None:
            none_dims.append(out_pos)
            out_pos += 1
            continue

        # Generate code based on the pattern type
        index_code = _pallas_generated_index_code(
            pattern, idx, state, tensor, i, tensor_dim, in_pipeline, pipeline_block_ids
        )
        parts.append(index_code)

        out_pos += 1
        tensor_dim += 1

    requries_smem_access = all(":" not in p and "pl.ds" not in p for p in parts)
    if requries_smem_access:
        state.codegen.device_function.pallas_smem_tensor_ids.add(id(tensor))

    return ", ".join(parts), none_dims


def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


def _pallas_get_indexing_patterns(
    state: CodegenState, tensor: torch.Tensor
) -> list[object]:
    assert state.fx_node is not None
    assert hasattr(state.fx_node, "meta")
    patterns = state.fx_node.meta.get("indexing_patterns")
    assert patterns is not None, f"No indexing patterns found for node {state.fx_node}"
    return patterns


def _pallas_generated_index_code(
    pattern: object,
    idx: object,
    state: CodegenState,
    tensor: torch.Tensor,
    subscript_index: int,
    tensor_dim: int,
    in_pipeline: bool,
    pipeline_block_ids: set[int],
) -> str:
    """Generate index code based on the indexing pattern."""
    from .._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from .._compiler.pallas.plan_tiling import ArbitrarySlicePattern
    from .._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from .._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from .._compiler.pallas.plan_tiling import TilePattern

    if isinstance(pattern, TilePattern):
        return _pallas_tile_pattern_code(
            pattern, idx, state, tensor_dim, in_pipeline, pipeline_block_ids
        )

    if isinstance(pattern, TileIndexWithOffsetPattern):
        return _pallas_tile_index_with_offset_pattern_code(pattern, state)

    if isinstance(pattern, TileBeginWithOffsetPattern):
        return _pallas_tile_begin_with_offset_pattern_code(
            pattern, state, subscript_index, tensor_dim
        )

    if isinstance(pattern, ArbitrarySlicePattern):
        return _pallas_slice_code(idx, pattern, state, tensor, tensor_dim)

    if isinstance(pattern, ArbitraryIndexPattern):
        if isinstance(idx, int):
            return str(idx)
        return _pallas_index_expr_from_ast(state, subscript_index)

    raise RuntimeError(
        f"Unhandled indexing pattern type: {type(pattern).__name__}. "
        f"Pattern: {pattern}, idx: {idx}, subscript_index: {subscript_index}. "
        f"All indexing patterns should be handled by the tiling analysis system."
    )


def _pallas_tile_pattern_code(
    pattern: object,
    idx: object,
    state: CodegenState,
    tensor_dim: int,
    in_pipeline: bool,
    pipeline_block_ids: set[int],
) -> str:
    from .._compiler.pallas.plan_tiling import TilePattern
    from .._compiler.tile_strategy import DeviceLoopState
    from .._compiler.tile_strategy import ForiLoopState

    assert isinstance(pattern, TilePattern)

    block_id = pattern.block_id

    can_tile = _can_tile_dimension(state, tensor_dim)
    if not can_tile:
        return _pallas_ds_expr(state, block_id)

    if in_pipeline and block_id in pipeline_block_ids:
        return ":"

    loops = state.codegen.active_device_loops.get(block_id)
    if loops and any(
        isinstance(loop, DeviceLoopState)
        or (isinstance(loop, ForiLoopState) and not loop.use_dma)
        for loop in loops
    ):
        return _pallas_ds_expr(state, block_id)
    return ":"


def _pallas_tile_index_with_offset_pattern_code(
    pattern: object,
    state: CodegenState,
) -> str:
    from .._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern

    assert isinstance(pattern, TileIndexWithOffsetPattern)

    block_id = pattern.block_id
    offset_str = f"{pattern.offset}"
    return _pallas_ds_expr(state, block_id, offset_str)


def _pallas_tile_begin_with_offset_pattern_code(
    pattern: object,
    state: CodegenState,
    subscript_index: int,
    tensor_dim: int,
) -> str:
    from .._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from .._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, TileBeginWithOffsetPattern)

    can_tile = _can_tile_dimension(state, tensor_dim)

    if not can_tile:
        return _pallas_index_expr_from_ast(state, subscript_index)

    assert isinstance(pattern.offset, int)

    loops = state.codegen.active_device_loops.get(pattern.block_id)
    if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
        offset = state.codegen.offset_var(pattern.block_id)
        if pattern.offset != 0:
            offset = f"{offset} + {pattern.offset}"
        return offset

    return f"{pattern.offset}"


def _pallas_index_expr_from_ast(state: CodegenState, subscript_index: int) -> str:
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[subscript_index]
    assert isinstance(ast_idx, ast.AST)
    name = state.codegen.lift(ast_idx, dce=True, prefix="index")
    return name.id


def _pallas_slice_code(
    idx: object,
    pattern: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> str:
    from .._compiler.pallas.plan_tiling import ArbitrarySlicePattern
    from .._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, ArbitrarySlicePattern)

    if idx != slice(None):
        raise AssertionError(
            f"Arbitrary slice expr {slice} not supported in Pallas backend yet"
        )

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor.shape[tensor_dim])
    if block_id is not None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
            if block_id is not None:
                return _pallas_ds_expr(state, block_id)

    return ":"


def _pallas_ds_expr(state: CodegenState, block_id: int, tile_offset: str = "") -> str:
    """Return a ``pl.ds(offset, block_size)`` expression for *block_id*, offset by *tile_offset*"""
    offset = state.codegen.offset_var(block_id)
    if tile_offset:
        offset = f"{offset} + {tile_offset}"
    block_size = state.device_function.block_size_var(block_id)
    if block_size is None:
        return ":"
    return f"pl.ds({offset}, {block_size})"


def _pallas_vmem_name(state: CodegenState, name: str) -> str:
    """Remap a tensor name to its VMEM ref name when inside emit_pipeline or fori_loop."""
    from .._compiler.tile_strategy import EmitPipelineLoopState
    from .._compiler.tile_strategy import ForiLoopState

    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, (EmitPipelineLoopState, ForiLoopState)):
                mapping = getattr(loop, "_tensor_to_vmem", None)
                if mapping and name in mapping:
                    return mapping[name]
    return name


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    name = _pallas_vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    index_str, _ = _pallas_index_str(state, subscript, tensor)
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {{value}}", value=value)
    )


def _matching_block_ids(env: CompileEnvironment, size: object) -> list[int]:
    """Find all block_ids that match the given dimension size."""
    candidates: list[int] = []
    if isinstance(size, (int, torch.SymInt)):
        if (direct := env.get_block_id(size)) is not None:
            candidates.append(direct)
    if not isinstance(size, (int, torch.SymInt)):
        return candidates
    for info in env.block_sizes:
        if not isinstance(info.size, (int, torch.SymInt)):
            continue
        if not env.known_equal(info.size, size):
            continue
        if info.block_id not in candidates:
            candidates.append(info.block_id)
    return candidates


def _log_cute_layout(state: CodegenState, op_name: str) -> None:
    """Log the CuTe layout annotation for the current node, if any.

    This is used during CuTe load/store codegen to make layout info
    visible for debugging and future codegen integration.
    """
    layout = state.cute_layout
    if layout is None:
        return
    node_name = state.fx_node.name if state.fx_node else "?"
    log.debug(
        "cute %s %s: layout tag=%s thread=%s value=%s",
        op_name,
        node_name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _cute_active_index_var(state: CodegenState, block_id: int) -> str | None:
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.index_var(block_id)
    grid_state = state.codegen.current_grid_state
    if grid_state is not None and block_id in grid_state.block_ids:
        return grid_state.strategy.index_var(block_id)
    return None


def _cute_active_mask_var(state: CodegenState, block_id: int) -> str | None:
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.mask_var(block_id)
    return None


def _cute_unique_graph_block_id(state: CodegenState) -> int | None:
    fx_node = state.fx_node
    if fx_node is None:
        return None
    graph_block_ids = [
        graph_info.block_ids
        for graph_info in state.codegen.codegen_graphs
        if graph_info.graph is fx_node.graph and hasattr(graph_info, "block_ids")
    ]
    if len(graph_block_ids) != 1 or len(graph_block_ids[0]) != 1:
        return None
    (block_id,) = graph_block_ids[0]
    return block_id


def _maybe_codegen_cute_packed_affine_lhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> object | None:
    from .._compiler.cute.indexing import CutePackedAffineLoad
    from .._compiler.cute.indexing import match_cute_affine_range_iota
    from .._compiler.cute.indexing import match_cute_stack_reshape_rhs
    from .matmul_ops import dot

    fx_node = state.fx_node
    if (
        fx_node is None
        or len(fx_node.users) != 1
        or len(subscript) not in (2, 3)
        or len(fx_node.args) < 2
    ):
        return None

    fx_subscript = fx_node.args[1]
    if not isinstance(fx_subscript, (list, tuple)) or len(fx_subscript) != len(
        subscript
    ):
        return None
    range_node = fx_subscript[-1]
    if not isinstance(range_node, torch.fx.Node):
        return None
    affine_range = match_cute_affine_range_iota(range_node)
    if affine_range is None:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target not in {
        dot,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }:
        return None

    rhs_index = (
        2
        if user.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default)
        else 1
    )
    rhs_arg = user.args[rhs_index]
    if not isinstance(rhs_arg, torch.fx.Node):
        return None
    packed_rhs = match_cute_stack_reshape_rhs(rhs_arg)
    if packed_rhs is None:
        return None
    _, factor = packed_rhs
    if factor != affine_range.factor:
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-1]]
    row_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(row_index_exprs) != len(leading_subscript):
        return None

    tensor_name = state.device_function.tensor_arg(tensor).name
    mask_terms: list[str] = []
    row_mask = _cute_combined_mask(state, leading_subscript, extra_mask, tensor=tensor)
    if row_mask is not None:
        mask_terms.append(row_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    mask_expr = " and ".join(mask_terms) if mask_terms else None
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    terms: list[ast.AST] = []
    for offset in range(factor):
        index_expr = ", ".join(
            [
                *row_index_exprs,
                f"cutlass.Int32({factor}) * ({packed_index}) + cutlass.Int32({offset})",
            ]
        )
        term = expr_from_string(f"{tensor_name}[{index_expr}]")
        if mask_expr is not None:
            term = expr_from_string(
                f"({{value}} if {mask_expr} else {zero}(0))",
                value=term,
            )
        terms.append(term)
    return CutePackedAffineLoad(tuple(terms))


def _maybe_codegen_cute_packed_rhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    from .._compiler.cute.indexing import match_cute_duplicate_stack_reshape_rhs

    fx_node = state.fx_node
    if fx_node is None or len(subscript) not in (2, 3) or len(fx_node.users) != 1:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target is not torch.ops.aten.stack.default:
        return None
    stack_users = list(user.users)
    if len(stack_users) != 1 or not isinstance(stack_users[0], torch.fx.Node):
        return None
    rhs_node = stack_users[0]
    packed_rhs = match_cute_duplicate_stack_reshape_rhs(rhs_node)
    if packed_rhs != (
        fx_node,
        len(user.args[0]) if isinstance(user.args[0], (list, tuple)) else 0,
    ):
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-2]]
    col_index_exprs = _cute_index_exprs(
        state,
        [subscript[-1]],
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(col_index_exprs) != 1:
        return None
    (col_index,) = col_index_exprs
    leading_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(leading_index_exprs) != len(leading_subscript):
        return None
    tensor_name = state.device_function.tensor_arg(tensor).name
    load_index_expr = ", ".join([*leading_index_exprs, packed_index, col_index])
    load_expr: ast.AST = expr_from_string(f"{tensor_name}[{load_index_expr}]")
    mask_terms: list[str] = []
    col_mask = _cute_combined_mask(
        state,
        [*leading_subscript, subscript[-1]],
        extra_mask,
        tensor=tensor,
    )
    if col_mask is not None:
        mask_terms.append(col_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    if not mask_terms:
        return load_expr
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(
        f"({{value}} if {' and '.join(mask_terms)} else {zero}(0))",
        value=load_expr,
    )


def _cute_index_exprs(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...] | None = None,
    tensor: torch.Tensor | None = None,
    *,
    inactive_slice_expr: str | None = None,
    inactive_singleton_slice_expr: str | None = None,
) -> list[str]:
    env = CompileEnvironment.current()

    def symint_index_expr(idx: torch.SymInt, used_block_ids: set[int]) -> str:
        expr = _symint_expr(idx)
        if expr is not None:
            origin_info = HostFunction.current().expr_to_origin.get(expr)
            if origin_info is not None and isinstance(origin_info.origin, GridOrigin):
                if type(origin_info.origin) is not GridOrigin:
                    block_id = origin_info.origin.block_id
                    loop_info = active_loop_info(block_id)
                    begin_var = tile_begin_expr(block_id, loop_info)
                    block_size_var = (
                        state.device_function.block_size_var(block_id) or "1"
                    )
                    if isinstance(origin_info.origin, TileBeginOrigin):
                        return begin_var
                    if isinstance(origin_info.origin, TileEndOrigin):
                        if loop_info is not None and loop_info.end_var_name is not None:
                            return loop_info.end_var_name
                        return f"({begin_var}) + ({block_size_var})"
                    if isinstance(origin_info.origin, TileCountOrigin):
                        end_var = (
                            loop_info.end_var_name
                            if loop_info is not None
                            and loop_info.end_var_name is not None
                            else f"({begin_var}) + ({block_size_var})"
                        )
                        extent = f"({end_var}) - ({begin_var})"
                        return env.backend.cdiv_expr(
                            extent, block_size_var, is_device=True
                        )
                    if isinstance(origin_info.origin, TileIdOrigin):
                        if block_size_var == "1":
                            return begin_var
                        return f"({begin_var}) // ({block_size_var})"
                    return state.sympy_expr(expr)
        block_id = env.get_block_id(idx)
        if block_id is not None:
            used_block_ids.add(block_id)
            return index_var_for_block_id(block_id, idx)
        if expr is not None:
            return state.sympy_expr(expr)
        raise exc.BackendUnsupported("cute", f"unlowerable symbolic index: {idx}")

    def active_loop_info(block_id: int) -> LoopDimInfo | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].block_id_to_info.get(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            return grid_state.block_id_to_info.get(block_id)
        return None

    def active_local_coord(block_id: int) -> str | None:
        from .._compiler.cute.cute_reshape import _grid_local_coord_expr

        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            thread_axis = loops[-1].block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        return None

    def tile_begin_expr(block_id: int, loop_info: LoopDimInfo | None) -> str:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
        begin_var = "0"
        if loop_info is not None and loop_info.begin_var_name is not None:
            begin_var = loop_info.begin_var_name
        global_index = active_index_var(block_id)
        local_coord = active_local_coord(block_id)
        if global_index is not None and local_coord is not None:
            return state.codegen.lift(
                expr_from_string(f"({global_index}) - ({local_coord})"),
                dce=True,
                prefix="tile_begin",
            ).id
        if global_index is not None:
            return global_index
        return begin_var

    def active_index_var(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def resolve_active_slice_block_id(
        size: object,
        used_block_ids: set[int],
    ) -> int | None:
        candidates = _matching_block_ids(env, size)
        active_candidates = [
            block_id
            for block_id in candidates
            if active_index_var(block_id) is not None
        ]
        active_unused_candidates = [
            block_id for block_id in active_candidates if block_id not in used_block_ids
        ]
        if len(active_unused_candidates) == 1:
            return active_unused_candidates[0]
        if len(active_candidates) == 1:
            return active_candidates[0]
        if len(active_unused_candidates) > 1:
            reduction_unused = [
                block_id
                for block_id in active_unused_candidates
                if env.block_sizes[block_id].reduction
            ]
            if len(reduction_unused) == 1:
                return reduction_unused[0]
        if len(active_candidates) > 1:
            reduction_active = [
                block_id
                for block_id in active_candidates
                if env.block_sizes[block_id].reduction
            ]
            if len(reduction_active) == 1:
                return reduction_active[0]
        return None

    def index_var_for_block_id(block_id: int, size: object) -> str:
        if (idx_var := active_index_var(block_id)) is not None:
            return idx_var

        raise exc.BackendUnsupported(
            "cute",
            (
                "indexing dimension is not active in this scope "
                f"(block_id={block_id}, size={size})"
            ),
        )

    used_block_ids = {
        block_id
        for idx in subscript
        if isinstance(idx, torch.SymInt)
        if (block_id := env.get_block_id(idx)) is not None
    }
    result = []
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        ast_idx = None
        if ast_subscript is not None:
            ast_idx = ast_subscript[pos]
        if idx is None:
            continue
        if isinstance(idx, torch.SymInt):
            result.append(symint_index_expr(idx, used_block_ids))
            tensor_dim += 1
        elif isinstance(idx, int):
            result.append(str(idx))
            tensor_dim += 1
        elif isinstance(idx, torch.Tensor):
            from .._compiler.cute.indexing import CuteAffineRangeIndex

            if isinstance(ast_idx, CuteAffineRangeIndex):
                raise exc.BackendUnsupported(
                    "cute",
                    "affine hl.arange() indexing is only supported in CuTe packed-matmul load fusion",
                )
            if not isinstance(ast_idx, ast.AST):
                raise exc.BackendUnsupported(
                    "cute", f"tensor index without AST at position {pos}"
                )
            lifted = state.codegen.lift(ast_idx, dce=True, prefix="index")
            index_dtype = env.backend.dtype_str(env.index_dtype)
            result.append(f"{index_dtype}({lifted.id})")
            tensor_dim += 1
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor is None:
                raise exc.BackendUnsupported("cute", "slice indexing without tensor")
            dim_size = tensor.shape[tensor_dim]
            block_id = resolve_active_slice_block_id(dim_size, used_block_ids)
            if block_id is not None:
                idx_var = active_index_var(block_id)
                assert idx_var is not None
                used_block_ids.add(block_id)
                result.append(idx_var)
                tensor_dim += 1
                continue
            if inactive_singleton_slice_expr is not None and env.known_equal(
                dim_size, 1
            ):
                result.append(inactive_singleton_slice_expr)
                tensor_dim += 1
                continue
            if inactive_slice_expr is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(tensor_dim={pos}, size={dim_size})"
                    ),
                )
            result.append(inactive_slice_expr)
            tensor_dim += 1
        else:
            raise exc.BackendUnsupported("cute", f"index type: {type(idx)}")
    return result


def _cute_index_tuple(index_exprs: list[str]) -> str:
    if len(index_exprs) == 1:
        return f"({index_exprs[0]},)"
    return f"({', '.join(index_exprs)})"


def _cute_combined_mask(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    tensor: torch.Tensor | None = None,
) -> str | None:
    env = CompileEnvironment.current()
    terms: list[str] = []

    def mask_var_for_block_id(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.mask_var(block_id)
        return None

    if extra_mask is not None:
        terms.append(state.codegen.lift(extra_mask, dce=True, prefix="mask").id)

    seen: set[int] = set()
    tensor_dim = 0
    for idx in subscript:
        block_id: int | None = None
        if idx is None:
            continue
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif isinstance(idx, slice) and idx == slice(None) and tensor is not None:
            for bid in _matching_block_ids(env, tensor.shape[tensor_dim]):
                if bid not in seen and mask_var_for_block_id(bid) is not None:
                    block_id = bid
                    break
        else:
            tensor_dim += 1
            continue
        if block_id is None or block_id in seen:
            tensor_dim += 1
            continue
        seen.add(block_id)
        if (mask_var := mask_var_for_block_id(block_id)) is not None:
            if mask_var not in terms:
                terms.append(mask_var)
        tensor_dim += 1

    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _codegen_cute_store_permute_lane_loops(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from .._compiler.cute.cute_reshape import _coords_from_flat_index
    from .._compiler.cute.cute_reshape import _flat_index_from_coords
    from .._compiler.cute.cute_reshape import _get_dim_local_coord
    from .._compiler.cute.cute_reshape import _get_tile_shape
    from .._compiler.cute.cute_reshape import _permute_reorders_active_dims
    from .._compiler.cute.cute_reshape import _shape_op_needs_materialization
    from .._compiler.cute.cute_reshape import _store_permute_info
    from .._compiler.generate_ast import GenerateAST
    from .._compiler.tile_strategy import DeviceGridState

    if not isinstance(state.codegen, GenerateAST):
        return None
    grid_state = state.codegen.current_grid_state
    if not isinstance(grid_state, DeviceGridState) or not grid_state.has_lane_loops():
        return None
    if _shape_op_needs_materialization(value_node):
        return None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    index_tuple = _cute_index_tuple(index_exprs)
    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    tensor_name = state.device_function.tensor_arg(tensor).name

    input_node: torch.fx.Node
    output_val = value_node.meta.get("val")
    read_flat: str
    input_shape: list[int]

    info = _store_permute_info(value_node)
    if info is not None:
        input_node, perm = info
        input_val = input_node.meta.get("val")
        if not isinstance(input_val, torch.Tensor) or not isinstance(
            output_val, torch.Tensor
        ):
            return None
        if not _permute_reorders_active_dims(state.codegen, input_val, perm):
            return None
        source_tensor_node = input_node.args[0] if input_node.args else None
        source_extra_mask = input_node.args[2] if len(input_node.args) > 2 else None
        if (
            input_node.op == "call_function"
            and input_node.target is load
            and isinstance(source_tensor_node, torch.fx.Node)
            and source_extra_mask is None
        ):
            source_tensor = source_tensor_node.meta.get("val")
            if isinstance(source_tensor, torch.Tensor):
                reordered_subscript = [
                    subscript[perm.index(i)] for i in range(len(perm))
                ]
                reordered_ast_subscript = (
                    [ast_subscript[perm.index(i)] for i in range(len(perm))]
                    if isinstance(ast_subscript, (list, tuple))
                    else None
                )
                source_index_exprs = _cute_index_exprs(
                    state,
                    reordered_subscript,
                    ast_subscript=reordered_ast_subscript,
                    tensor=source_tensor,
                    inactive_singleton_slice_expr="0",
                )
                source_index_tuple = _cute_index_tuple(source_index_exprs)
                source_name = state.device_function.tensor_arg(source_tensor).name
                source_mask = _cute_combined_mask(
                    state,
                    reordered_subscript,
                    None,
                    tensor=source_tensor,
                )
                source_dtype = CompileEnvironment.current().backend.dtype_str(
                    source_tensor.dtype
                )
                return expr_from_string(
                    (
                        f"({tensor_name}.__setitem__({index_tuple}, "
                        f"({source_name}[{source_index_tuple}] if {source_mask} else {source_dtype}(0))) "
                        f"if {mask_expr} else None)"
                    )
                    if source_mask is not None and mask_expr is not None
                    else (
                        f"{tensor_name}.__setitem__({index_tuple}, "
                        f"{source_name}[{source_index_tuple}] if {source_mask} else {source_dtype}(0))"
                        if source_mask is not None
                        else (
                            f"({tensor_name}.__setitem__({index_tuple}, {source_name}[{source_index_tuple}]) "
                            f"if {mask_expr} else None)"
                            if mask_expr is not None
                            else f"{tensor_name}.__setitem__({index_tuple}, {source_name}[{source_index_tuple}])"
                        )
                    )
                )
            raise exc.BackendUnsupported("cute", "permute lane-loop source tensor")
        env = CompileEnvironment.current()
        df = state.device_function
        input_shape = _get_tile_shape(input_val, env, df.config)
        output_shape = _get_tile_shape(output_val, env, df.config)
        src_coords = [
            _get_dim_local_coord(state.codegen, input_val, i)
            for i in range(len(input_shape))
        ]
        current_flat = _flat_index_from_coords(src_coords, input_shape)
        output_coords = _coords_from_flat_index(current_flat, output_shape)
        read_coords = [output_coords[perm.index(i)] for i in range(len(perm))]
        read_flat = _flat_index_from_coords(read_coords, input_shape)
    elif value_node.target in {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    }:
        input_arg = value_node.args[0]
        if not isinstance(input_arg, torch.fx.Node):
            return None
        input_node = input_arg
        input_val = input_node.meta.get("val")
        if not isinstance(input_val, torch.Tensor) or not isinstance(
            output_val, torch.Tensor
        ):
            return None
        env = CompileEnvironment.current()
        df = state.device_function
        input_shape = _get_tile_shape(input_val, env, df.config)
        output_shape = _get_tile_shape(output_val, env, df.config)
        if input_shape == output_shape:
            return None
        input_non_unit = [s for s in input_shape if s != 1]
        output_non_unit = [s for s in output_shape if s != 1]
        if input_non_unit == output_non_unit:
            return None
        src_coords = [
            _get_dim_local_coord(state.codegen, input_val, i)
            for i in range(len(input_shape))
        ]
        current_flat = _flat_index_from_coords(src_coords, input_shape)
        output_coords = [
            _get_dim_local_coord(state.codegen, output_val, i)
            for i in range(len(output_shape))
        ]
        read_flat = _flat_index_from_coords(output_coords, output_shape)
    else:
        return None

    env = CompileEnvironment.current()
    df = state.device_function
    input_numel = 1
    for size in input_shape:
        input_numel *= size

    dtype_str = env.backend.dtype_str(input_val.dtype)
    smem_ptr = df.new_var("permute_smem_ptr")
    smem = df.new_var("permute_smem")
    state.codegen.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    state.codegen.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )

    read_expr = (
        f"{df.tensor_arg(tensor).name}.__setitem__({index_tuple}, {smem}[{read_flat}])"
        if mask_expr is None
        else (
            f"({df.tensor_arg(tensor).name}.__setitem__({index_tuple}, {smem}[{read_flat}]) "
            f"if {mask_expr} else None)"
        )
    )
    return expr_from_string(
        f"({smem}.__setitem__({current_flat}, {{value}}), "
        f"cute.arch.sync_threads(), "
        f"{read_expr})",
        value=value,
    )


@_decorators.codegen(store, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.store(ptr + offset, val, mask) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_store_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(state, tensor, [*subscript], value, extra_mask)
    raise exc.BackendUnsupported("metal", f"store target type: {type(tensor)}")


@_decorators.codegen(store, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if state.fx_node is not None and len(state.fx_node.args) > 2:
        value_node = state.fx_node.args[2]
        if isinstance(value_node, torch.fx.Node) and value_node.op == "call_function":
            if isinstance(tensor, torch.Tensor):
                rewritten_stmt = _codegen_cute_store_permute_lane_loops(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
            from .._compiler.cute.cute_reshape import codegen_cute_store_permute

            rewritten = codegen_cute_store_permute(state, value, value_node)
            if rewritten is not None:
                value = rewritten

    if isinstance(tensor, tuple):
        raise exc.BackendUnsupported("cute", "stack tensor store")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"store target type: {type(tensor)}")

    _log_cute_layout(state, "store")

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    index_tuple = _cute_index_tuple(index_exprs)
    assign_expr = expr_from_string(
        f"{tensor_name}.__setitem__({index_tuple}, {{value}})", value=value
    )

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if mask_expr is None:
        return assign_expr
    return expr_from_string(
        f"({tensor_name}.__setitem__({index_tuple}, {{value}}) if {mask_expr} else None)",
        value=value,
    )


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    from .ref_tile import RefTile

    # Normalize indices and identify tensor indices
    indices = []
    tensor_idx_positions = []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index
        # pyrefly: ignore [bad-argument-type]
        indices.append(idx)
        if isinstance(idx, torch.Tensor):
            tensor_idx_positions.append(i)

    # Handle broadcasting for multiple tensor indices
    if len(tensor_idx_positions) > 1:
        grids = torch.meshgrid(
            # pyrefly: ignore [bad-argument-type]
            *(indices[i] for i in tensor_idx_positions),
            indexing="ij",
        )
        for i, grid in zip(tensor_idx_positions, grids, strict=False):
            # pyrefly: ignore [unsupported-operation]
            indices[i] = grid

    if extra_mask is not None:
        mask = extra_mask.to(torch.bool)

        # Check bounds for tensor indices
        for i, idx in enumerate(indices):
            if isinstance(idx, torch.Tensor):
                mask = mask & (idx >= 0) & (idx < tensor.shape[i])
        mask_count = int(mask.sum().item())
        if mask_count == 0:
            return

        # Use index_put_ for masked stores
        valid_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                valid_indices.append(idx[mask].long())
            else:
                idx_val = int(idx) if isinstance(idx, torch.SymInt) else idx
                valid_indices.append(
                    # pyrefly: ignore [no-matching-overload]
                    torch.full(
                        (mask_count,), idx_val, dtype=torch.long, device=tensor.device
                    )
                )

        if isinstance(value, torch.Tensor):
            values = value[mask]
        else:
            val = int(value) if isinstance(value, torch.SymInt) else value
            values = torch.full(
                (mask_count,), val, dtype=tensor.dtype, device=tensor.device
            )

        # Check for duplicate indices - this is undefined behavior in Triton
        if valid_indices:
            stacked = torch.stack(valid_indices, dim=1)
            unique_count = stacked.unique(dim=0).size(0)
            if unique_count < stacked.size(0):
                raise exc.DuplicateStoreIndicesError(
                    "hl.store with duplicate indices has undefined behavior in compiled mode. "
                    "The order in which values are written to the same memory location is "
                    "non-deterministic and may vary between Triton versions and backends."
                )

        tensor.index_put_(tuple(valid_indices), values, accumulate=False)
        return

    # Simple assignment
    tensor[tuple(indices)] = (  # pyrefly: ignore[unsupported-operation]
        int(value) if isinstance(value, torch.SymInt) else value
    )


@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range. It also accepts an optional
    `eviction_policy` which is forwarded to the underlying Triton `tl.load`
    call to control the cache eviction behavior (e.g., "evict_last").

    Args:
        tensor: The tensor / stack tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
        eviction_policy: Optional Triton load eviction policy to hint cache behavior
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(load)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> tuple[torch.Tensor | tuple, list[object], torch.Tensor | None, str | None]:
    from .tile_proxy import Tile

    index = Tile._tiles_to_sizes_for_index(index)
    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, extra_mask, eviction_policy)
    assert isinstance(tensor, torch.Tensor)
    return (tensor, index, extra_mask, eviction_policy)


@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        target_shape = SubscriptIndexing.compute_shape(tensor, index)
        env = CompileEnvironment.current()
        return env.new_index_result(tensor, target_shape)
    if isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        return tensor_like.new_empty(target_shape)
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


@_decorators.codegen(load, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function
    load_idx = device_fn.device_load_index
    device_fn.device_load_index += 1

    # If no explicit eviction_policy and we're in device code, use tunable
    if eviction_policy is None and state.codegen.on_device:
        policies = state.config.load_eviction_policies
        if load_idx < len(policies):
            policy_value = policies[load_idx]
            eviction_policy = _EVICTION_POLICY_MAP.get(policy_value, policy_value)

    if eviction_policy is not None:
        assert isinstance(eviction_policy, str)
        eviction_policy = ast.Constant(value=eviction_policy)

    if isinstance(tensor, torch.Tensor):
        # If tile_index(...) is being broadcast-only indexed
        from ..language import tile_index

        tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
        if (
            isinstance(tensor_node, torch.fx.Node)
            and tensor_node.op == "call_function"
            and tensor_node.target == tile_index
        ):
            # tile.index tensors are not real memory accesses; materialize the
            # block index variable with the requested broadcast/reshape.
            env = CompileEnvironment.current()
            block_id = env.get_block_id(tensor.size(0))
            assert block_id is not None
            base_var = state.codegen.index_var(block_id)

            parts = []
            for idx in subscript:
                if idx is None:
                    parts.append("None")
                elif idx == slice(None):
                    parts.append(":")
                else:
                    raise AssertionError(
                        f"Unexpected index type in tile_index load: {idx}"
                    )
            return expr_from_string(f"{base_var}[{', '.join(parts)}]")

        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        if state.codegen.load_transform is not None:
            return state.codegen.load_transform(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                strategy.codegen_load,
            )

        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy
        )
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack loads (multi-tensor device pointers);
        # fall through to the unfused path regardless of load_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state, tensor, dev_ptrs_ast, [*subscript], extra_mask, eviction_policy
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


@_decorators.codegen(load, "pallas")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    name = state.device_function.tensor_arg(tensor).name
    name = _pallas_vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_load_index += 1
    device_fn.device_memory_op_index += 1
    index_str, none_dims = _pallas_index_str(state, subscript, tensor)
    result = expr_from_string(f"{name}[{index_str}]")
    for dim in none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


@_decorators.codegen(load, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal delegates to the same PointerIndexingStrategy as Triton.
    # This produces tl.load(ptr + offset, mask, other=0) in the AST;
    # the MSL walker translates it to Metal.
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None
    assert isinstance(eviction_policy, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_load_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy
        )
    raise exc.BackendUnsupported("metal", f"load tensor type: {type(tensor)}")


@_decorators.codegen(load, "cute")
def _(state: CodegenState) -> object:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, tuple):
        raise exc.BackendUnsupported("cute", "stack tensor load")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"load tensor type: {type(tensor)}")

    _log_cute_layout(state, "load")

    packed_affine_lhs = _maybe_codegen_cute_packed_affine_lhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_affine_lhs is not None:
        return packed_affine_lhs

    packed_rhs_load = _maybe_codegen_cute_packed_rhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_rhs_load is not None:
        return packed_rhs_load

    tensor_name = state.device_function.tensor_arg(tensor).name
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    load_expr = f"{tensor_name}[{', '.join(index_exprs)}]"
    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0


# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(load)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    from .ref_tile import RefTile

    if extra_mask is None:
        # Convert RefTiles to indices
        indices = [idx.index if isinstance(idx, RefTile) else idx for idx in index]
        # Use meshgrid for Cartesian product when we have multiple tensor indices
        tensor_idxs = [
            i for i, idx in enumerate(indices) if isinstance(idx, torch.Tensor)
        ]
        if len(tensor_idxs) > 1:
            # pyrefly: ignore [bad-argument-type]
            grids = torch.meshgrid(*(indices[i] for i in tensor_idxs), indexing="ij")
            for i, grid in zip(tensor_idxs, grids, strict=False):
                indices[i] = grid
        # pyrefly: ignore [bad-argument-type]
        return tensor[tuple(indices)]

    # Create zero result matching mask shape
    result = torch.zeros(extra_mask.shape, dtype=tensor.dtype, device=tensor.device)

    # Process indices: convert RefTiles and clamp tensor indices
    orig_indices, safe_indices, is_tensor_mask = [], [], []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index  # Convert RefTile to tensor

        if isinstance(idx, torch.Tensor):
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            orig_indices.append(idx)
            safe_indices.append(torch.clamp(idx, 0, dim_size - 1))
            is_tensor_mask.append(True)
        else:
            orig_indices.append(idx)
            safe_indices.append(idx)
            is_tensor_mask.append(False)

    # Apply broadcasting if we have multiple tensor indices
    tensor_positions = [i for i, is_tensor in enumerate(is_tensor_mask) if is_tensor]

    if len(tensor_positions) > 1:
        # Add unsqueeze operations for broadcasting
        broadcast_indices = []
        for i, (idx, is_tensor) in enumerate(
            zip(safe_indices, is_tensor_mask, strict=False)
        ):
            if is_tensor:
                new_idx = idx
                # Add dimension for each other tensor index
                for j, other_pos in enumerate(tensor_positions):
                    if other_pos != i:
                        new_idx = new_idx.unsqueeze(j if other_pos < i else -1)
                broadcast_indices.append(new_idx)
            else:
                broadcast_indices.append(idx)
        values = tensor[tuple(broadcast_indices)]
    else:
        values = tensor[tuple(safe_indices)]

    # Build validity mask
    valid_mask = extra_mask.clone()
    for i, (orig_idx, is_tensor) in enumerate(
        zip(orig_indices, is_tensor_mask, strict=False)
    ):
        if is_tensor:
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            in_bounds = (orig_idx >= 0) & (orig_idx < dim_size)
            # Broadcast to match mask shape by adding dimensions
            # Count how many tensor indices come before and after this one
            n_before = sum(1 for j in range(i) if is_tensor_mask[j])
            n_after = sum(
                1 for j in range(i + 1, len(is_tensor_mask)) if is_tensor_mask[j]
            )

            # Add dimensions: n_after dimensions at the end, n_before at the beginning
            for _ in range(n_after):
                in_bounds = in_bounds.unsqueeze(-1)
            for _ in range(n_before):
                in_bounds = in_bounds.unsqueeze(0)
            valid_mask = valid_mask & in_bounds

    return torch.where(valid_mask, values, result)
