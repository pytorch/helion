from __future__ import annotations

import ast
import contextlib
import logging
import operator
import textwrap
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect
from torch.fx.node import map_arg

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _symint_expr
from .._compiler.cute.cutedsl_compat import emit_dealloc_mbarrier_initialized_kwarg
from .._compiler.cute.cutedsl_compat import emit_pipeline_advance
from .._compiler.cute.cutedsl_compat import emit_producer_tail_tma_umma
from .._compiler.cute.cutedsl_compat import emit_producer_tail_umma_async
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.pallas import codegen as pallas_codegen
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


def _record_pad_info(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    block_id: int,
) -> None:
    """Record that a tensor dimension uses pl.ds() and may need host-side padding.

    Note: stores one block_id per (tensor, dim).  If two inner loops tile the
    same dim with different block_ids, the last one wins.  This is fine when
    both loops use the same block size (the common case).
    """
    pad_info = state.device_function.pallas_pad_info
    tensor_id = id(tensor)
    if tensor_id not in pad_info:
        pad_info[tensor_id] = {}
    pad_info[tensor_id][tensor_dim] = block_id


def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


@_decorators.codegen(store, "pallas")
def _(state: CodegenState) -> None:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    name = state.device_function.tensor_arg(tensor).name
    name = pallas_codegen.vmem_name(state, name)
    # Increment memory op index to stay in sync with triton backend
    device_fn = state.device_function
    device_fn.device_store_index += 1
    device_fn.device_memory_op_index += 1
    index_str, _ = pallas_codegen.index_str(state, subscript, tensor)
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
                            return env.backend.minimum_expr(
                                f"({begin_var}) + ({block_size_var})",
                                loop_info.end_var_name,
                            )
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
        if (
            tensor is not None
            and tensor_dim < tensor.ndim
            and env.known_equal(tensor.shape[tensor_dim], 1)
            and not (isinstance(idx, slice) and idx == slice(None))
        ):
            result.append("0")
            tensor_dim += 1
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


def _cute_affine_range_block_id(state: CodegenState, affine: object) -> int | None:
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    env = CompileEnvironment.current()
    base_meta = getattr(affine.base, "meta", {})
    base_val = base_meta.get("val") if isinstance(base_meta, dict) else None
    block_id = env.resolve_block_id(base_val) if base_val is not None else None
    if block_id is None:
        codegen = base_meta.get("codegen") if isinstance(base_meta, dict) else None
        if isinstance(codegen, ast.Name) and codegen.id.startswith("_BLOCK_SIZE_"):
            with contextlib.suppress(ValueError):
                block_id = int(codegen.id.removeprefix("_BLOCK_SIZE_"))
    if block_id is None:
        return None
    if state.fx_node is not None:
        return env.resolve_codegen_block_id(
            block_id, state.codegen, state.fx_node.graph
        )
    return block_id


def _cute_affine_range_expr(
    state: CodegenState,
    affine: object,
    lane_var: str,
    *,
    dtype: torch.dtype | None = None,
) -> str | None:
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        return None
    expr = f"({affine.factor}) * ({index_var}) + cutlass.Int32({lane_var})"
    if dtype is not None:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr


def _codegen_cute_affine_range_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from .._compiler.ast_extension import create
    from .._compiler.cute.indexing import CuteAffineRangeIndex

    affine_positions = [
        (pos, idx)
        for pos, idx in enumerate(ast_subscript)
        if isinstance(idx, CuteAffineRangeIndex)
    ]
    if len(affine_positions) != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    _pos, affine = affine_positions[0]
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None

    lane_var = state.device_function.new_var("affine_lane", dce=True)
    index_expr = _cute_affine_range_expr(
        state, affine, lane_var, dtype=CompileEnvironment.current().index_dtype
    )
    if index_expr is None:
        return None
    backend = CompileEnvironment.current().backend
    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor):
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        ast_source_subscript = list(
            map_arg(tuple(source_subscript), lambda arg: state.env[arg])
        )
        (source_affine,) = ast_source_subscript
        if not isinstance(source_affine, CuteAffineRangeIndex):
            return None
        if source_affine.factor != affine.factor:
            return None
        source_index_expr = _cute_affine_range_expr(
            state,
            source_affine,
            lane_var,
            dtype=CompileEnvironment.current().index_dtype,
        )
        if source_index_expr is None:
            return None
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index_expr}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, CuteAffineRangeIndex):
        value_expr = _cute_affine_range_expr(state, value, lane_var, dtype=value.dtype)
        if value_expr is None:
            return None
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = (
        f"{tensor_name}.__setitem__({_cute_index_tuple([index_expr])}, {value_expr})"
    )
    mask_var = _cute_active_mask_var(state, block_id)
    if mask_var is not None:
        store_expr = f"{store_expr} if {mask_var} else None"

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({affine.factor})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _is_cute_affine_range_load_for_store(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
) -> bool:
    from .._compiler.cute.indexing import CuteAffineRangeIndex
    from .._compiler.cute.indexing import match_cute_affine_range_iota

    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and isinstance(store_subscript[0], torch.fx.Node)
            and match_cute_affine_range_iota(store_subscript[0]) is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and len(subscript) == 1
        and len(ast_subscript) == 1
        and isinstance(ast_subscript[0], CuteAffineRangeIndex)
    )


def _cute_positive_1d_slice_bounds(
    tensor: torch.Tensor, index: object
) -> tuple[int, int, int, int] | None:
    if not isinstance(index, slice) or index == slice(None):
        return None
    with contextlib.suppress(TypeError):
        dim_size = int(tensor.shape[0])
        start, stop, step = index.indices(dim_size)
        if step <= 0:
            return None
        length = max(0, (stop - start + step - 1) // step)
        return start, stop, step, length
    return None


def _is_cute_strided_slice_load_for_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        target_node = user.args[0]
        if not isinstance(target_node, torch.fx.Node):
            return False
        target_tensor = target_node.meta.get("val")
        if not isinstance(target_tensor, torch.Tensor) or target_tensor.ndim != 1:
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and _cute_positive_1d_slice_bounds(target_tensor, store_subscript[0])
            is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and tensor.ndim == 1
        and len(subscript) == 1
        and _cute_positive_1d_slice_bounds(tensor, subscript[0]) is not None
    )


def _codegen_cute_strided_slice_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from .._compiler.ast_extension import create

    if tensor.ndim != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    target_bounds = _cute_positive_1d_slice_bounds(tensor, subscript[0])
    if target_bounds is None:
        return None
    target_start, _target_stop, target_step, target_length = target_bounds

    env = CompileEnvironment.current()
    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    loop_var = state.device_function.new_var("slice_idx", dce=True)
    target_index = f"{index_dtype}({target_start} + {loop_var} * {target_step})"

    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor) or source_tensor.ndim != 1:
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_bounds = _cute_positive_1d_slice_bounds(
            source_tensor, source_subscript[0]
        )
        if source_bounds is None:
            return None
        source_start, _source_stop, source_step, source_length = source_bounds
        if source_length != target_length:
            return None
        source_index = f"{index_dtype}({source_start} + {loop_var} * {source_step})"
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_name = state.device_function.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    store_expr = f"{target_name}.__setitem__(({target_index},), {value_expr})"
    return create(
        ast.For,
        target=create(ast.Name, id=loop_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({target_length})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


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
        elif isinstance(idx, torch.Tensor):
            for dim_size in idx.shape:
                for bid in _matching_block_ids(env, dim_size):
                    if bid in seen:
                        continue
                    mask_var = mask_var_for_block_id(bid)
                    if mask_var is not None:
                        seen.add(bid)
                        if mask_var not in terms:
                            terms.append(mask_var)
                        break
                else:
                    continue
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


def _cute_tensor_dim_size_expr(
    state: CodegenState, tensor: torch.Tensor, dim: int
) -> str:
    return state.device_function.tensor_size(tensor, dim).name


def _cute_tile_begin_expr(state: CodegenState, idx: object) -> str:
    env = CompileEnvironment.current()

    def active_index_var(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
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

    def tile_begin_from_block_id(block_id: int) -> str:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
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
        return "0"

    if isinstance(idx, int):
        return str(idx)
    if not isinstance(idx, torch.SymInt):
        raise exc.BackendUnsupported("cute", f"tile base index type: {type(idx)}")

    expr = _symint_expr(idx)
    if expr is not None:
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is not None and isinstance(origin_info.origin, TileBeginOrigin):
            return tile_begin_from_block_id(origin_info.origin.block_id)
    block_id = env.get_block_id(idx)
    if block_id is not None:
        return tile_begin_from_block_id(block_id)
    if expr is not None:
        return state.sympy_expr(expr)
    raise exc.BackendUnsupported("cute", f"unlowerable tile base index: {idx}")


def _codegen_cute_store_tcgen05_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_name: str,
) -> list[ast.AST] | ast.AST | None:
    if extra_mask is not None or tensor.ndim != 2:
        return None

    tcgen05_value = state.device_function.get_cute_tcgen05_store_value(value_name)
    if tcgen05_value is None:
        return None

    # Backstop for callers that bypass Config.normalize() validation;
    # see _tcgen05_epi_warp_count docstring and cute_plan.md.
    if tcgen05_value.epi_warp_count != 4:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 SIMT-store epilogue requires "
            f"tcgen05_num_epi_warps=4 (got {tcgen05_value.epi_warp_count}). "
            "CUTLASS tmem_warp_shape_mn=(4,1) hard-codes a 4-warp t2r "
            "partition for the supported tcgen05 path; per-warp "
            "tcgen05.ld semantics make the partition uncoverable by "
            "fewer warps. Lifts when the c_pipeline-driven multi-warp "
            "epilogue lands (see cute_plan.md).",
        )

    backend = CompileEnvironment.current().backend
    df = state.device_function
    tensor_name = df.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    base_indices = [_cute_tile_begin_expr(state, idx) for idx in subscript]
    if len(base_indices) != 2:
        return None

    m_size = _cute_tensor_dim_size_expr(state, tensor, 0)
    n_size = _cute_tensor_dim_size_expr(state, tensor, 1)
    tile_coord_m = f"({base_indices[0]}) // cutlass.Int32({tcgen05_value.bm})"
    tile_coord_n = f"({base_indices[1]}) // cutlass.Int32({tcgen05_value.bn})"
    full_tile = df.new_var("tcgen05_full_tile")

    gmem_tile = df.new_var("tcgen05_gC")
    coord_tile = df.new_var("tcgen05_cC")
    tcgc_base = df.new_var("tcgen05_tCgC_base")
    tccc_base = df.new_var("tcgen05_tCcC_base")
    tcgc = df.new_var("tcgen05_tCgC")
    tcgc_planned = df.new_var("tcgen05_tCgC_planned")
    tccc = df.new_var("tcgen05_tCcC")
    tacc = df.new_var("tcgen05_tAcc")
    epi_tile = df.new_var("tcgen05_store_epi_tile")
    tiled_copy_t2r = df.new_var("tcgen05_tiled_copy_t2r")
    thr_copy_t2r = df.new_var("tcgen05_thr_copy_t2r")
    ttr_tacc_base = df.new_var("tcgen05_tTR_tAcc_base")
    tcgc_epi = df.new_var("tcgen05_tCgC_epi")
    tccc_epi = df.new_var("tcgen05_tCcC_epi")
    ttr_gc = df.new_var("tcgen05_tTR_gC")
    ttr_cc = df.new_var("tcgen05_tTR_cC")
    ttr_racc = df.new_var("tcgen05_tTR_rAcc")
    ttr_rd = df.new_var("tcgen05_tTR_rD")
    ttr_tacc_stage = df.new_var("tcgen05_tTR_tAcc_stage")
    ttr_tacc = df.new_var("tcgen05_tTR_tAcc")
    ttr_gc_grouped = df.new_var("tcgen05_tTR_gC_grouped")
    ttr_cc_grouped = df.new_var("tcgen05_tTR_cC_grouped")
    ttr_tacc_mn = df.new_var("tcgen05_tTR_tAcc_mn")
    ttr_gc_subtile = df.new_var("tcgen05_tTR_gC_subtile")
    ttr_cc_subtile = df.new_var("tcgen05_tTR_cC_subtile")
    pred_c = df.new_var("tcgen05_pred_C")
    pred_c_shape = df.new_var("tcgen05_pred_C_shape")
    acc_vec = df.new_var("tcgen05_acc_vec")
    kernel_desc = df.new_var("tcgen05_kernel_desc")
    mcld = df.new_var("tcgen05_mcld")
    num_bits = df.new_var("tcgen05_num_bits")
    simt_atom = df.new_var("tcgen05_simt_atom")
    smem_d_layout = df.new_var("tcgen05_sD_layout")
    smem_d_ptr = df.new_var("tcgen05_sD_ptr")
    smem_d = df.new_var("tcgen05_sD")
    tiled_copy_r2s = df.new_var("tcgen05_tiled_copy_r2s")
    trs_rd = df.new_var("tcgen05_tRS_rD")
    trs_racc = df.new_var("tcgen05_tRS_rAcc")
    trs_sd = df.new_var("tcgen05_tRS_sD")
    bsg_sd = df.new_var("tcgen05_bSG_sD")
    bsg_gd_partitioned = df.new_var("tcgen05_bSG_gD_partitioned")
    bsg_gd = df.new_var("tcgen05_bSG_gD")
    c_buffer = df.new_var("tcgen05_c_buffer")
    epilog_sync_barrier = df.new_var("tcgen05_epilog_sync_barrier")
    c_pipeline_producer_group = df.new_var("tcgen05_c_pipeline_producer_group")
    c_pipeline = df.new_var("tcgen05_c_pipeline")
    subtile_count = df.new_var("tcgen05_subtile_count")
    epi_warp_ids = ", ".join(
        f"cutlass.Int32({i})" for i in range(tcgen05_value.epi_warp_count)
    )
    if tcgen05_value.epi_warp_count == 1:
        epi_warp_ids += ","
    if tcgen05_value.use_tma_store_epilogue:
        df.placeholder_args.add(tensor_name)
        df.wrapper_only_params.extend(
            [tcgen05_value.tma_store_atom, tcgen05_value.tma_store_tensor]
        )
        state.codegen.cute_wrapper_plans.append(
            {
                "kind": "tcgen05_d_tma",
                "d_name": tensor_name,
                "bm": tcgen05_value.bm,
                "bn": tcgen05_value.bn,
                "c_stage_count": tcgen05_value.c_stage_count,
                "output_dtype": target_dtype,
                "kernel_args": [
                    tcgen05_value.tma_store_atom,
                    tcgen05_value.tma_store_tensor,
                ],
            }
        )

    tcgen05_bm = tcgen05_value.bm
    tcgen05_bn = tcgen05_value.bn
    tcgen05_bk = tcgen05_value.bk
    tcgen05_epilog_sync_barrier_id = tcgen05_value.epilog_sync_barrier_id
    tcgen05_c_stage_count = tcgen05_value.c_stage_count
    tcgen05_is_two_cta = tcgen05_value.is_two_cta
    tcgen05_thr_mma = tcgen05_value.thr_mma

    def store_common_setup(gmem_tensor: str, *, include_full_tile: bool) -> list[str]:
        setup = [
            (
                f"{kernel_desc} = type('Tcgen05KernelDesc', (), {{"
                f"'cta_tile_shape_mnk': ({tcgen05_bm}, {tcgen05_bn}, {tcgen05_bk}), "
                "'c_layout': cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                f"'c_dtype': {target_dtype}, "
                "'acc_dtype': cutlass.Float32, "
                f"'epilog_sync_bar_id': cutlass.Int32({tcgen05_epilog_sync_barrier_id}), "
                f"'epilogue_warp_id': ({epi_warp_ids}), "
                f"'num_c_stage': cutlass.Int32({tcgen05_c_stage_count}), "
                f"'use_2cta_instrs': {tcgen05_is_two_cta!s}"
                "})()"
            ),
            (
                f"{epi_tile} = cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape("
                f"({tcgen05_bm}, {tcgen05_bn}), False, "
                f"cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {target_dtype})"
            ),
        ]
        if include_full_tile:
            setup.append(
                f"{full_tile} = "
                f"({base_indices[0]}) + cutlass.Int32({tcgen05_bm}) <= {m_size} "
                f"and ({base_indices[1]}) + cutlass.Int32({tcgen05_bn}) <= {n_size}"
            )
        setup.extend(
            [
                (
                    f"{gmem_tile} = cute.local_tile("
                    f"{gmem_tensor}, ({tcgen05_bm}, {tcgen05_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                ),
                f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
            ]
        )
        return setup

    simt_common_store_setup = store_common_setup(tensor_name, include_full_tile=True)
    tma_common_store_setup = store_common_setup(
        tcgen05_value.tma_store_tensor, include_full_tile=False
    )
    simt_store_body_core = [
        *simt_common_store_setup,
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
        (
            f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
            f"{kernel_desc}, {tcgen05_value.epi_tidx}, {tacc}, {tcgc_planned}, {epi_tile}, {tcgen05_value.is_two_cta!s})"
        ),
        f"{thr_copy_t2r} = {tiled_copy_t2r}.get_slice({tcgen05_value.epi_tidx})",
        f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        f"{ttr_gc} = {thr_copy_t2r}.partition_D({tcgc_epi})",
        (
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_value.acc_consumer_state}.index)]"
        ),
        (
            f"if {tcgen05_value.epi_active}:\n"
            f"    {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})"
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{ttr_gc_grouped} = cute.group_modes({ttr_gc}, 3, cute.rank({ttr_gc}))",
        (
            f"{ttr_racc} = cute.make_rmem_tensor("
            f"{ttr_gc_grouped}[(None, None, None, 0)].shape, cutlass.Float32)"
        ),
        f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})",
        (
            f"{mcld} = cute.max_common_layout("
            f"{ttr_rd}.layout, {ttr_gc_grouped}[(None, None, None, 0)].layout)"
        ),
        (
            f"{num_bits} = min("
            f"{ttr_gc_grouped}.iterator.alignment * 8, "
            f"cute.size({mcld}) * {target_dtype}.width, 256)"
        ),
        (
            f"{simt_atom} = cute.make_copy_atom("
            f"cute.nvgpu.CopyUniversalOp(), {target_dtype}, "
            f"num_bits_per_copy={num_bits}, "
            f"l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)"
        ),
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        (
            # Per-subtile loop: TMEM->reg (t2r) first, then reg->GMEM (SIMT
            # store). On the last subtile we release the acc consumer slot
            # *before* the GMEM store so the next mainloop tile's MMA can
            # producer_acquire the TMEM stage and begin issuing UMMAs while
            # this tile's epilogue is still draining to GMEM. This mirrors the
            # release-acc-inside-the-subtile-loop pattern in Quack's sm100
            # gemm epilogue. Without c_pipeline SMEM staging we can only
            # release after the final t2r (not per-subtile), but even one
            # tile of overlap measurably improves the wide tcgen05 path on
            # B200. `cutlass.range(..., unroll_full=True)` keeps the loop
            # statically unrolled so `tiled_copy_t2r` (a TiledCopy that wraps
            # a tcgen05 tmem_load atom) is not captured as an scf.for iter_arg
            # — the cute-to-nvvm pass cannot legalize that conversion through
            # iter_args and aborts during compile.
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"    if {tcgen05_value.epi_active}:\n"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        {ttr_gc_subtile} = {ttr_gc_grouped}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            f"        {acc_vec} = {ttr_racc}.load().to({target_dtype})\n"
            f"        {ttr_rd}.store({acc_vec})\n"
            f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
            # `cute.copy(t2r, ...)` issues async TMEM->reg loads. Releasing
            # the acc consumer slot lets the MMA producer re-acquire the TMEM
            # stage and issue UMMAs that overwrite TMEM, so we must fence the
            # in-flight async TMEM loads first to avoid a race on the last
            # subtile's `ttr_racc` / `ttr_rd` data. This matches Quack's
            # sm100 gemm fence-before-release pattern.
            f"            cute.arch.fence_view_async_tmem_load()\n"
            f"            with cute.arch.elect_one():\n"
            f"                {tcgen05_value.acc_pipeline}.consumer_release({tcgen05_value.acc_consumer_state})\n"
            + emit_pipeline_advance(
                tcgen05_value.acc_consumer_state, indent="            "
            )
            + "\n"
            f"        if {full_tile}:\n"
            f"            cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile})\n"
            f"        else:\n"
            f"            {coord_tile} = cute.local_tile(cute.make_identity_tensor(({m_size}, {n_size})), ({tcgen05_value.bm}, {tcgen05_value.bn}), ({tile_coord_m}, {tile_coord_n}))\n"
            f"            {tccc_base} = {tcgen05_value.thr_mma}.partition_C({coord_tile})\n"
            f"            {tccc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({tccc_base})\n"
            f"            {tccc_epi} = cute.flat_divide({tccc}, {epi_tile})\n"
            f"            {ttr_cc} = {thr_copy_t2r}.partition_D({tccc_epi})\n"
            f"            {ttr_cc_grouped} = cute.group_modes({ttr_cc}, 3, cute.rank({ttr_cc}))\n"
            f"            {ttr_cc_subtile} = {ttr_cc_grouped}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"            {pred_c_shape} = (1, *{ttr_cc_subtile}.shape[1:])\n"
            f"            {pred_c} = cute.make_rmem_tensor({pred_c_shape}, cutlass.Boolean)\n"
            f"            for _pred_m in range({ttr_cc_subtile}.shape[1]):\n"
            f"                for _pred_n in range({ttr_cc_subtile}.shape[2]):\n"
            f"                    _coord = {ttr_cc_subtile}[(0, _pred_m, _pred_n)]\n"
            f"                    {pred_c}[(0, _pred_m, _pred_n)] = cute.elem_less(_coord, ({m_size}, {n_size}))\n"
            f"            cute.copy({simt_atom}, {ttr_rd}, {ttr_gc_subtile}, pred={pred_c})\n"
        ),
    ]
    tma_store_body_core = [
        *tma_common_store_setup,
        # Must match the wrapper-side `tcgen05_d_tma` TMA atom layout in
        # `helion/runtime/__init__.py`; both describe one D SMEM stage.
        (
            f"{smem_d_layout} = cutlass.utils.blackwell_helpers.make_smem_layout_epi("
            f"{target_dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
            f"{epi_tile}, {tcgen05_value.c_stage_count})"
        ),
        (
            f"{smem_d_ptr} = cute.arch.alloc_smem("
            f"{target_dtype}, cute.cosize({smem_d_layout}.outer), alignment=1024)"
        ),
        (
            f"{smem_d} = cute.make_tensor("
            f"cute.recast_ptr({smem_d_ptr}, {smem_d_layout}.inner, dtype={target_dtype}), "
            f"{smem_d_layout}.outer)"
        ),
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
        (
            f"{tiled_copy_t2r}, {ttr_tacc_base}, {ttr_racc} = "
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition("
            f"{kernel_desc}, {tcgen05_value.epi_tidx}, {tacc}, {tcgc_planned}, {epi_tile}, {tcgen05_value.is_two_cta!s})"
        ),
        (f"{ttr_rd} = cute.make_rmem_tensor({ttr_racc}.shape, {target_dtype})"),
        (
            f"{tiled_copy_r2s}, {trs_rd}, {trs_sd} = "
            "cutlass.utils.gemm.sm100.epilogue_smem_copy_and_partition("
            f"{kernel_desc}, {tiled_copy_t2r}, {ttr_rd}, "
            f"{tcgen05_value.epi_tidx}, {smem_d})"
        ),
        f"{trs_racc} = {tiled_copy_r2s}.retile({ttr_racc})",
        f"{tcgc_epi} = cute.flat_divide({tcgc_planned}, {epi_tile})",
        (
            f"{bsg_sd}, {bsg_gd_partitioned} = cute.nvgpu.cpasync.tma_partition("
            f"{tcgen05_value.tma_store_atom}, 0, cute.make_layout(1), "
            f"cute.group_modes({smem_d}, 0, 2), "
            f"cute.group_modes({tcgc_epi}, 0, 2))"
        ),
        (
            f"{bsg_gd} = {bsg_gd_partitioned}["
            f"(None, None, None, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))]"
        ),
        f"{bsg_gd} = cute.group_modes({bsg_gd}, 1, cute.rank({bsg_gd}))",
        (
            f"{epilog_sync_barrier} = cutlass.pipeline.NamedBarrier("
            f"barrier_id=cutlass.Int32({tcgen05_value.epilog_sync_barrier_id}), "
            f"num_threads=cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        (
            f"{c_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
            f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_value.epi_warp_count * 32}))"
        ),
        (
            f"{c_pipeline} = cutlass.pipeline.PipelineTmaStore.create("
            f"num_stages={tcgen05_value.c_stage_count}, "
            f"producer_group={c_pipeline_producer_group})"
        ),
        (
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_value.acc_consumer_state}.index)]"
        ),
        (
            f"if {tcgen05_value.epi_active}:\n"
            f"    {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})"
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        (
            # Per-subtile loop: all epi warps stage TMEM->reg->SMEM, then a
            # CTA-scoped named barrier makes the SMEM stage visible before
            # warp 0 issues the S2G TMA copy. A second barrier keeps the epi
            # warps from reusing that SMEM stage before warp 0 has committed
            # the TMA operation. `PipelineTmaStore.create` starts with stages
            # available, so the first subtile intentionally has no explicit
            # acquire before its copy. The current acquire-after-commit order
            # is correct but may serialize some next-subtile work; investigate
            # acquire-before-R2S staging as a follow-up before changing it.
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"    if {tcgen05_value.epi_active}:\n"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            f"        {acc_vec} = {trs_racc}.load().to({target_dtype})\n"
            f"        {trs_rd}.store({acc_vec})\n"
            f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
            f"            cute.arch.fence_view_async_tmem_load()\n"
            f"            with cute.arch.elect_one():\n"
            f"                {tcgen05_value.acc_pipeline}.consumer_release({tcgen05_value.acc_consumer_state})\n"
            + emit_pipeline_advance(
                tcgen05_value.acc_consumer_state, indent="            "
            )
            + "\n"
            f"        {c_buffer} = cutlass.Int32(_tcgen05_subtile) % cutlass.Int32({tcgen05_value.c_stage_count})\n"
            f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
            f"        cute.arch.fence_proxy('async.shared', space='cta')\n"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            cute.copy({tcgen05_value.tma_store_atom}, {bsg_sd}[(None, {c_buffer})], {bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            f"            {c_pipeline}.producer_commit()\n"
            f"            {c_pipeline}.producer_acquire()\n"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
        ),
        (
            f"if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"    {c_pipeline}.producer_tail()"
        ),
    ]
    store_body_core = (
        tma_store_body_core
        if tcgen05_value.use_tma_store_epilogue
        else simt_store_body_core
    )
    main_stmts: list[ast.AST]
    if tcgen05_value.use_role_local_epi:
        sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body_core), "    ")
        )
        sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
        df.register_cute_tcgen05_per_tile_stmts(
            [sync_before_stmt, main_stmt, sync_after_stmt]
        )
        df.register_cute_tcgen05_epi_role_stmts([main_stmt])
        main_stmts = [sync_before_stmt, main_stmt, sync_after_stmt]
    else:
        store_body = [
            "cute.arch.sync_threads()",
            *store_body_core,
            "cute.arch.sync_threads()",
        ]
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body), "    ")
        )
        main_stmts = [main_stmt]
    # Pipeline drain + TMEM dealloc are one-shot cleanup. They must run
    # AFTER all tiles have been processed (in the persistent path) and
    # naturally land at the end of the kernel in the non-persistent path.
    # Keep them as separate statements so the persistent splitter can
    # extract them via the post-loop registration below.
    post_loop_lines: list[str] = []
    if tcgen05_value.use_tma:
        post_loop_lines.append(
            f"if {tcgen05_value.tma_warp}:\n"
            + emit_producer_tail_tma_umma(
                tcgen05_value.tma_pipeline,
                tcgen05_value.tma_producer_state,
                num_stages=tcgen05_value.ab_stage_count,
                indent="    ",
            )
        )
    post_loop_lines.extend(
        [
            (
                f"if {tcgen05_value.exec_active}:\n"
                f"    {tcgen05_value.tmem_alloc_barrier}.arrive()"
            ),
            (
                f"if {tcgen05_value.exec_active}:\n"
                + emit_producer_tail_umma_async(
                    tcgen05_value.acc_pipeline,
                    tcgen05_value.acc_producer_state,
                    num_stages=tcgen05_value.acc_stage_count,
                    indent="    ",
                )
            ),
            (
                f"{tcgen05_value.tmem_allocator} = cutlass.utils.TmemAllocator("
                f"{tcgen05_value.tmem_holding_buf}, "
                f"barrier_for_retrieve={tcgen05_value.tmem_alloc_barrier}, "
                f"allocator_warp_id=0, is_two_cta={tcgen05_value.is_two_cta!s}, "
                f"two_cta_tmem_dealloc_mbar_ptr={tcgen05_value.tmem_dealloc_mbar_ptr}, "
                f"num_allocated_columns={tcgen05_value.acc_tmem_cols}"
                f"{emit_dealloc_mbarrier_initialized_kwarg()})"
            ),
            "cute.arch.sync_threads()",
            (
                f"if {tcgen05_value.epi_active}:\n"
                f"    {tcgen05_value.tmem_allocator}.relinquish_alloc_permit()"
            ),
            (
                f"if {tcgen05_value.epi_active}:\n"
                f"    {tcgen05_value.tmem_alloc_barrier}.arrive_and_wait()"
            ),
            (
                f"if {tcgen05_value.epi_active}:\n"
                f"    {tcgen05_value.tmem_allocator}.free({tcgen05_value.epi_acc_tmem_ptr})"
            ),
        ]
    )
    post_loop_stmts: list[ast.AST] = [
        statement_from_string(line) for line in post_loop_lines
    ]
    df.register_cute_tcgen05_post_loop_stmts(post_loop_stmts)
    return [*main_stmts, *post_loop_stmts]


def _codegen_cute_store_loaded_index_trailing_slices(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from .._compiler.ast_extension import create

    if value_node.target is not load or len(value_node.args) < 2:
        return None
    source_tensor_node = value_node.args[0]
    if not isinstance(source_tensor_node, torch.fx.Node):
        return None
    source_tensor = source_tensor_node.meta.get("val")
    if not isinstance(source_tensor, torch.Tensor):
        return None
    source_subscript = value_node.args[1]
    if not isinstance(source_subscript, (list, tuple)) or not source_subscript:
        return None
    indexer = source_subscript[0]
    if not isinstance(indexer, torch.fx.Node):
        return None
    indexer_value = indexer.meta.get("val")
    if not isinstance(indexer_value, torch.Tensor) or indexer_value.ndim == 0:
        return None
    trailing_source = [*source_subscript[1:]]
    if not trailing_source or not all(idx == slice(None) for idx in trailing_source):
        return None
    if len(subscript) != indexer_value.ndim + len(trailing_source):
        return None
    trailing_store = subscript[indexer_value.ndim :]
    if not all(idx == slice(None) for idx in trailing_store):
        return None

    ast_source_subscript = list(
        map_arg(tuple(source_subscript), lambda arg: state.env[arg])
    )
    index_exprs = _cute_index_exprs(
        state,
        [indexer_value],
        [ast_source_subscript[0]],
        tensor=source_tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != 1:
        return None

    prefix_subscript = [*subscript[: indexer_value.ndim]]
    prefix_ast_subscript = [*ast_subscript[: indexer_value.ndim]]
    target_prefix = _cute_index_exprs(
        state,
        prefix_subscript,
        prefix_ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(target_prefix) != indexer_value.ndim:
        return None

    env = CompileEnvironment.current()
    index_dtype = env.backend.dtype_str(env.index_dtype)
    source_loop_vars = [
        state.device_function.new_var("slice_idx", dce=True) for _ in trailing_source
    ]
    source_indices = [
        index_exprs[0],
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    target_indices = [
        *target_prefix,
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    if len(source_indices) != source_tensor.ndim or len(target_indices) != tensor.ndim:
        return None

    source_name = state.device_function.tensor_arg(source_tensor).name
    target_name = state.device_function.tensor_arg(tensor).name
    source_dtype = env.backend.dtype_str(source_tensor.dtype)
    target_dtype = env.backend.dtype_str(tensor.dtype)
    source_mask = _cute_combined_mask(
        state,
        [indexer_value],
        None,
        tensor=source_tensor,
    )
    target_mask = _cute_combined_mask(
        state,
        prefix_subscript,
        extra_mask,
        tensor=tensor,
    )
    masks = [mask for mask in (source_mask, target_mask) if mask is not None]
    mask_expr = " and ".join(f"({mask})" for mask in masks) if masks else None
    load_expr = f"{source_name}[{', '.join(source_indices)}]"
    if mask_expr is not None:
        load_expr = f"({load_expr} if {mask_expr} else {source_dtype}(0))"
    store_expr = (
        f"{target_name}.__setitem__({_cute_index_tuple(target_indices)}, "
        f"{env.backend.ast_to_dtype_expr(load_expr, target_dtype)})"
    )
    if mask_expr is not None:
        store_expr = f"{store_expr} if {mask_expr} else None"

    tensor_dim = 0
    for idx in prefix_subscript:
        block_id = None
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif idx == slice(None) and tensor_dim < tensor.ndim:
            block_id = next(
                (
                    candidate
                    for candidate in _matching_block_ids(env, tensor.shape[tensor_dim])
                    if candidate in state.codegen.active_device_loops
                ),
                None,
            )
        tensor_dim += 1
        if block_id is None:
            continue
        axis = None
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            axis = grid_state.block_thread_axes.get(block_id)
        if axis is None:
            loops = state.codegen.active_device_loops.get(block_id)
            if loops:
                axis = loops[-1].block_thread_axes.get(block_id)
        if axis is None or not (0 <= axis < 3):
            continue
        block_size = env.block_sizes[block_id].from_config(state.config)
        if not isinstance(block_size, int):
            continue
        state.codegen.max_thread_block_dims[axis] = max(
            state.codegen.max_thread_block_dims[axis],
            block_size,
        )
        state.codegen.referenced_thread_block_dims[axis] = max(
            state.codegen.referenced_thread_block_dims[axis],
            block_size,
        )

    stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    for loop_var, source_pos in reversed(
        [*zip(source_loop_vars, range(1, len(source_subscript)), strict=True)]
    ):
        extent = _cute_tensor_dim_size_expr(state, source_tensor, source_pos)
        stmt = create(
            ast.For,
            target=create(ast.Name, id=loop_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent})"),
            body=[stmt],
            orelse=[],
            type_comment=None,
        )
    state.add_statement(stmt)
    return ast.Constant(value=None)


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
    raw_value = state.ast_args[2]
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    value_node = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        maybe_value_node = state.fx_node.args[2]
        if isinstance(maybe_value_node, torch.fx.Node):
            value_node = maybe_value_node

    if isinstance(tensor, torch.Tensor):
        affine_range_store = _codegen_cute_affine_range_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if affine_range_store is not None:
            state.add_statement(affine_range_store)
            return ast.Constant(value=None)
        strided_slice_store = _codegen_cute_strided_slice_store(
            state,
            tensor,
            subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if strided_slice_store is not None:
            state.add_statement(strided_slice_store)
            return ast.Constant(value=None)

    value = state.ast_arg(2)

    if value_node is not None:
        if value_node.op == "call_function":
            if isinstance(tensor, torch.Tensor):
                rewritten_stmt = _codegen_cute_store_loaded_index_trailing_slices(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
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

    if isinstance(value, ast.Name):
        rewritten_stmt = _codegen_cute_store_tcgen05_tile(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value.id,
        )
        if rewritten_stmt is not None:
            stmts = (
                rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
            )
            for stmt in stmts:
                state.add_statement(stmt)
            return ast.Constant(value=None)

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
    topk_lane_expr: object | None = None
    topk_k: object | None = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        value_node = state.fx_node.args[2]
        if (
            isinstance(value_node, torch.fx.Node)
            and value_node.target is operator.getitem
            and isinstance(value_node.args[0], torch.fx.Node)
            and value_node.args[0].target is torch.ops.aten.topk.default
        ):
            topk_lane_expr = value_node.args[0].meta.get("cute_topk_lane_expr")
            topk_k = value_node.args[0].meta.get("cute_topk_k")
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        index_exprs[-1] = topk_lane_expr
    index_tuple = _cute_index_tuple(index_exprs)
    assign_expr = expr_from_string(
        f"{tensor_name}.__setitem__({index_tuple}, {{value}})", value=value
    )

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        topk_mask = f"({topk_lane_expr}) < {topk_k}"
        mask_expr = topk_mask if mask_expr is None else f"({mask_expr}) and {topk_mask}"
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
        env.backend.process_fake_tensor_load(tensor, index)
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
    return pallas_codegen.load_expr(state, list(subscript), tensor)


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

    from ..language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(tensor.size(0))
        if block_id is None:
            raise exc.BackendUnsupported("cute", "tile_index load block id")
        index_var = _cute_active_index_var(state, block_id)
        if index_var is None:
            raise exc.BackendUnsupported("cute", "inactive tile_index load")
        for idx in subscript:
            if idx is None or idx == slice(None):
                continue
            raise exc.BackendUnsupported(
                "cute", f"tile_index load index type: {type(idx)}"
            )
        return expr_from_string(index_var)

    if state.device_function.suppress_cute_root_lane_loops or (
        state.fx_node is not None
        and state.device_function.is_cute_collective_handled_load(state.fx_node.name)
    ):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

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

    if _is_cute_affine_range_load_for_store(state, subscript, ast_subscript):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")
    if _is_cute_strided_slice_load_for_store(state, tensor, subscript):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

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
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if state.fx_node is not None and any(
        user.target in (torch.ops.aten.sort.default, torch.ops.aten.topk.default)
        or getattr(user.target, "__name__", None) == "_associative_scan"
        for user in state.fx_node.users
    ):
        from .._compiler.cute.indexing import CuteSortableLoad

        tensor_dim = 0
        sort_index_pos = -1
        for idx in subscript:
            if idx is None:
                continue
            if tensor_dim == tensor.ndim - 1:
                sort_index_pos = tensor_dim
                break
            tensor_dim += 1
        if sort_index_pos < 0:
            raise exc.BackendUnsupported("cute", "sort/topk input rank")
        sortable_load = CuteSortableLoad(
            expr=expr_from_string(
                load_expr
                if mask_expr is None
                else f"({load_expr} if {mask_expr} else {CompileEnvironment.current().backend.dtype_str(tensor.dtype)}(0))"
            ),
            tensor_name=tensor_name,
            index_exprs=tuple(index_exprs),
            sort_index_pos=sort_index_pos,
            mask_expr=mask_expr,
            dtype=tensor.dtype,
        )
        state.fx_node.meta["cute_sortable_load"] = sortable_load
        return sortable_load.expr
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
        # pyrefly: ignore [bad-argument-type, bad-index]
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
