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
from .._compiler.cute.cute_epilogue import Tcgen05UnaryEpilogueChain
from .._compiler.cute.cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .._compiler.cute.cute_fx_walk import reach_tcgen05_matmul_anchors
from .._compiler.cute.cutedsl_compat import emit_dealloc_mbarrier_initialized_kwarg
from .._compiler.cute.cutedsl_compat import emit_pipeline_advance
from .._compiler.cute.cutedsl_compat import emit_producer_tail_tma_umma
from .._compiler.cute.cutedsl_compat import emit_producer_tail_umma_async
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R,
)
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.indexing_strategy import TileWithOffsetInfo
from .._compiler.indexing_strategy import _get_tile_with_offset_info
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
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


def _record_pad_info(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    block_id: int,
    extra_pad: int = 0,
) -> None:
    """Record that a tensor dimension uses pl.ds() and may need host-side padding.

    *extra_pad* accounts for non-zero loop begins: 0 when the loop starts
    at offset 0, ``begin % block_size`` for a constant begin, or
    ``block_size - 1`` for a data-dependent begin.

    Note: stores one entry per (tensor, dim).  If two inner loops tile the
    same dim with different block_ids, the last one wins.  This is fine when
    both loops use the same block size (the common case).
    """
    pad_info = state.device_function.pallas_pad_info
    tensor_id = id(tensor)
    if tensor_id not in pad_info:
        pad_info[tensor_id] = {}
    pad_info[tensor_id][tensor_dim] = (block_id, extra_pad)


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

    def local_coord_for_block_id(block_id: int, begin_var: str) -> str | None:
        if (local_coord := active_local_coord(block_id)) is not None:
            return local_coord
        if (idx_var := active_index_var(block_id)) is not None:
            return f"({idx_var}) - ({begin_var})"
        return None

    def tile_with_offset_index_expr(tile_info: TileWithOffsetInfo) -> str:
        block_id = tile_info.block_id
        begin_var = tile_begin_expr(block_id, active_loop_info(block_id))
        local_coord = local_coord_for_block_id(block_id, begin_var)
        if local_coord is None:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "indexing dimension is not active in this scope "
                    f"(block_id={block_id})"
                ),
            )
        offset_expr = state.device_function.literal_expr(tile_info.offset)
        return f"({begin_var}) + cutlass.Int32({offset_expr}) + ({local_coord})"

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
        if (
            tile_info := _get_tile_with_offset_info(
                idx, getattr(state, "fx_node", None), pos
            )
        ) is not None and tile_info.block_size is not None:
            used_block_ids.add(tile_info.block_id)
            result.append(tile_with_offset_index_expr(tile_info))
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


def _cute_scalar_pointer_expr(tensor_name: str, index_exprs: list[str]) -> str:
    env = CompileEnvironment.current()
    index_dtype = env.index_type()
    offset = " + ".join(
        f"({index_dtype}({index}) * {index_dtype}({tensor_name}.layout.stride[{dim}]))"
        for dim, index in enumerate(index_exprs)
    )
    return f"({tensor_name}.iterator + {offset})"


def _cute_scalar_load_expr(tensor_name: str, index_exprs: list[str]) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}[{', '.join(index_exprs)}]"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.load()"


def _cute_scalar_store_expr(
    tensor_name: str, index_exprs: list[str], value: str
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}.__setitem__({_cute_index_tuple(index_exprs)}, {value})"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.store({value})"


def _cute_stack_tensor_offset_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object] | tuple[object, ...],
) -> str:
    env = CompileEnvironment.current()
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor_like,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in index_exprs:
        raise exc.BackendUnsupported("cute", "inactive stack tensor load dimension")
    index_dtype = env.index_type()
    terms = []
    for dim, index in enumerate(index_exprs):
        stride = tensor_like.stride(dim)
        stride_expr = (
            str(stride) if isinstance(stride, int) else state.sympy_expr(stride)
        )
        terms.append(f"({index_dtype}({index}) * {index_dtype}({stride_expr}))")
    return " + ".join(terms) if terms else "0"


def _cute_stack_tensor_mask_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
) -> str | None:
    terms = []
    tensor_mask = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor_like,
        include_tensor_index_masks=False,
    )
    if tensor_mask is not None:
        terms.append(tensor_mask)
    stack_mask = _cute_combined_mask(
        state,
        [slice(None)] * dev_ptrs.ndim,
        None,
        tensor=dev_ptrs,
    )
    if stack_mask is not None and stack_mask not in terms:
        terms.append(stack_mask)
    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_stack_tensor_pointer_expr(
    target_dtype: str,
    dev_ptrs_ast: ast.AST,
    offset_expr: str,
) -> ast.AST:
    return expr_from_string(
        f"(cute.make_ptr({target_dtype}, cutlass.Int64({{base}}), "
        f"cute.AddressSpace.gmem) + ({offset_expr}))",
        base=dev_ptrs_ast,
    )


def _codegen_cute_store_stack_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: tuple[object, ...] | list[object],
    ast_subscript: tuple[object, ...] | list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    if value_node.op != "call_function" or value_node.target is not load:
        return None
    stack_arg = value_node.args[0]
    if not isinstance(stack_arg, tuple) or len(stack_arg) != 2:
        return None
    ptr_node = stack_arg[1]
    if (
        not isinstance(ptr_node, torch.fx.Node)
        or ptr_node.op != "call_function"
        or ptr_node.target is not load
        or len(ptr_node.args) < 2
    ):
        return None
    dev_ptrs = (
        ptr_node.args[0].meta.get("val")
        if isinstance(ptr_node.args[0], torch.fx.Node)
        else None
    )
    ptr_subscript = ptr_node.args[1]
    if not isinstance(dev_ptrs, torch.Tensor) or not isinstance(
        ptr_subscript, (list, tuple)
    ):
        return None
    tensor_like_node = stack_arg[0]
    tensor_like = (
        tensor_like_node.meta.get("val")
        if isinstance(tensor_like_node, torch.fx.Node)
        else tensor_like_node
    )
    if not isinstance(tensor_like, torch.Tensor):
        return None

    if (
        dev_ptrs.ndim == 2
        and len(ptr_subscript) == 2
        and all(isinstance(idx, slice) and idx == slice(None) for idx in ptr_subscript)
        and len(subscript) >= 3
        and isinstance(subscript[0], slice)
        and subscript[0] == slice(None)
        and isinstance(subscript[1], slice)
        and subscript[1] == slice(None)
    ):
        stack_value_subscript = value_node.args[1]
        if not isinstance(stack_value_subscript, (list, tuple)):
            return None
        stack_value_subscript_proxy = map_arg(
            stack_value_subscript, lambda arg: arg.meta["val"]
        )
        stack_value_subscript_ast = map_arg(
            stack_value_subscript, lambda arg: state.env[arg]
        )
        tensor_offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*stack_value_subscript_proxy],
            [*stack_value_subscript_ast],
        )
        target_index_exprs = _cute_index_exprs(
            state,
            [*subscript],
            ast_subscript,
            tensor=tensor,
            inactive_singleton_slice_expr="0",
        )
        if len(target_index_exprs) != tensor.ndim:
            return None
        first_stack_index = target_index_exprs[0]
        target_tail = target_index_exprs[2:]
        loop_var = state.device_function.new_var("stack_dim", dce=True)
        env = CompileEnvironment.current()
        index_dtype = env.index_type()
        dev_ptrs_name = state.device_function.tensor_arg(dev_ptrs).name
        tensor_name = state.device_function.tensor_arg(tensor).name
        target_dtype = env.backend.dtype_str(tensor.dtype)
        dev_ptr_offset = (
            f"{index_dtype}({first_stack_index}) * "
            f"{index_dtype}({dev_ptrs.stride(0)}) + "
            f"{index_dtype}({loop_var}) * {index_dtype}({dev_ptrs.stride(1)})"
        )
        stack_ptr_expr = (
            f"(cute.make_ptr({target_dtype}, "
            f"cutlass.Int64(({dev_ptrs_name}.iterator + {dev_ptr_offset}).load()), "
            f"cute.AddressSpace.gmem) + ({tensor_offset_expr}))"
        )
        target_indices = [first_stack_index, loop_var, *target_tail]
        store_expr = _cute_scalar_store_expr(
            tensor_name,
            target_indices,
            f"({stack_ptr_expr}).load()",
        )
        mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
        if mask_expr is None:
            body = f"    {store_expr}"
        else:
            body = f"    if {mask_expr}:\n        {store_expr}"
        state.add_statement(
            statement_from_string(
                f"for {loop_var} in range({dev_ptrs.size(1)}):\n{body}"
            )
        )
        return ast.Constant(value=None)

    ptr_subscript_proxy = map_arg(ptr_subscript, lambda arg: arg.meta["val"])
    ptr_subscript_ast = map_arg(ptr_subscript, lambda arg: state.env[arg])
    ptr_index_exprs = _cute_index_exprs(
        state,
        [*ptr_subscript_proxy],
        [*ptr_subscript_ast],
        tensor=dev_ptrs,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in ptr_index_exprs:
        return None

    target_index_exprs = _cute_index_exprs(
        state,
        [*subscript],
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    ptr_pos = 0
    rewritten_index_exprs = []
    for idx, index_expr in zip(subscript, target_index_exprs, strict=True):
        if isinstance(idx, slice) and idx == slice(None):
            replacement = (
                ptr_index_exprs[ptr_pos] if ptr_pos < len(ptr_index_exprs) else None
            )
            ptr_pos += 1
            rewritten_index_exprs.append(
                replacement if replacement is not None else index_expr
            )
        else:
            if ptr_pos < len(ptr_subscript_proxy) and not (
                isinstance(ptr_subscript_proxy[ptr_pos], slice)
                and ptr_subscript_proxy[ptr_pos] == slice(None)
            ):
                ptr_pos += 1
            rewritten_index_exprs.append(index_expr)

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, rewritten_index_exprs, "{value}"),
        value=value,
    )
    mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
    if mask_expr is None:
        return store_expr
    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    state.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=mask_ast,
                body=[ast.Expr(value=store_expr)],
                orelse=[],
            )
        )
    )
    return ast.Constant(value=None)


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
    *,
    include_tensor_index_masks: bool = True,
) -> str | None:
    env = CompileEnvironment.current()
    terms: list[str] = []

    def mask_var_for_block_id(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.mask_var(block_id)
        return None

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

    def tile_begin_expr(block_id: int) -> str:
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

    def tile_with_offset_mask_terms(
        tile_info: TileWithOffsetInfo,
        tensor_dim: int,
    ) -> list[str]:
        block_id = tile_info.block_id
        local_coord = active_local_coord(block_id)
        begin_var = tile_begin_expr(block_id)
        if local_coord is None:
            if (idx_var := active_index_var(block_id)) is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(block_id={block_id})"
                    ),
                )
            local_coord = f"({idx_var}) - ({begin_var})"

        tile_terms = []
        if tile_info.block_size is not None:
            block_size_expr = state.device_function.literal_expr(tile_info.block_size)
            tile_terms.append(f"({local_coord}) < cutlass.Int32({block_size_expr})")
        if tensor is not None and tensor_dim < tensor.ndim:
            offset_expr = state.device_function.literal_expr(tile_info.offset)
            dim_size = _cute_tensor_dim_size_expr(state, tensor, tensor_dim)
            tile_terms.append(
                f"(({begin_var}) + cutlass.Int32({offset_expr}) + "
                f"({local_coord})) < {dim_size}"
            )
        return tile_terms

    if extra_mask is not None:
        terms.append(state.codegen.lift(extra_mask, dce=True, prefix="mask").id)

    seen: set[int] = set()
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        block_id: int | None = None
        if idx is None:
            continue
        if (
            tile_info := _get_tile_with_offset_info(
                idx, getattr(state, "fx_node", None), pos
            )
        ) is not None and tile_info.block_size is not None:
            seen.add(tile_info.block_id)
            for term in tile_with_offset_mask_terms(tile_info, tensor_dim):
                if term not in terms:
                    terms.append(term)
            tensor_dim += 1
            continue
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif isinstance(idx, slice) and idx == slice(None) and tensor is not None:
            for bid in _matching_block_ids(env, tensor.shape[tensor_dim]):
                if bid not in seen and mask_var_for_block_id(bid) is not None:
                    block_id = bid
                    break
        elif isinstance(idx, torch.Tensor):
            if not include_tensor_index_masks:
                for dim_size in idx.shape:
                    for bid in _matching_block_ids(env, dim_size):
                        if bid in seen or not env.is_jagged_tile(bid):
                            continue
                        mask_var = mask_var_for_block_id(bid)
                        if mask_var is not None:
                            seen.add(bid)
                            if mask_var not in terms:
                                terms.append(mask_var)
                            break
                tensor_dim += 1
                continue
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
            tensor_dim += 1
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
    epilogue_chain: Tcgen05UnaryEpilogueChain | None = None,
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
    # The matmul plan computed `tcgen05_epi_tile` (role-local t2r
    # partition) with `epi_elem_dtype_str`; the store path below
    # recomputes `tcgen05_store_epi_tile` with `target_dtype`. They must
    # match or `compute_epilogue_tile_shape` selects different `tile_n`
    # values on the two sides and the t2r / r2s SMEM staging silently
    # corrupts. The loud-failure backstop covers cases where MMA-codegen-
    # time forward-tracing of the matmul fx_node could not pin a unique
    # store target dtype.
    if (
        tcgen05_value.epi_elem_dtype_str
        and tcgen05_value.epi_elem_dtype_str != target_dtype
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 epilogue element-type mismatch: matmul plan was set "
            f"up with epi_elem_dtype_str={tcgen05_value.epi_elem_dtype_str!r} "
            f"but the store target tensor dtype is {target_dtype!r}.",
        )
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

    # G3.1.1: render the per-thread carrier expression for the
    # accumulator vector. The identity epilogue (no chain or empty
    # chain) emits the original `rAcc.load().to(target_dtype)` line.
    # When a whitelisted unary chain is present, hoist `rAcc.load()` to
    # a local TensorSSA so a chain like `cute.where(x > 0, x, 0)` reads
    # the loaded vector once. Each splice site below uses the
    # appropriate carrier name (`ttr_racc` for the SIMT path,
    # `trs_racc` for the TMA path, and `tcgen05_tRS_rAcc` for the
    # @cute.jit module helper). The returned snippet is a sequence of
    # zero-or-one prelude statements (each newline-terminated, indented
    # with `prelude_indent`) plus the assignment expression for
    # `tcgen05_acc_vec`.
    def _splice_acc_vec(carrier_name: str, prelude_indent: str) -> tuple[str, str]:
        """Return ``(prelude, assignment_rhs)``. ``prelude`` is empty
        for the identity epilogue. ``assignment_rhs`` is the right-hand
        side of ``acc_vec = ...`` (without leading whitespace or the
        trailing newline).

        Each chain step renders into a fresh ``tcgen05_chain_step*``
        local so chain composition stays linear in source size — the
        relu template duplicates ``{inner}`` 5 times, so without per-
        step binding a 3-deep relu chain would emit 125x duplication
        and pessimize parse / IR-build time. Per-step locals keep
        the rendered source O(N) in chain depth and CuTe CSEs the
        loads at compile.
        """
        load_expr = f"{carrier_name}.load()"
        if epilogue_chain is None or not epilogue_chain.steps:
            return ("", f"{load_expr}.to({target_dtype})")
        loaded = df.new_var("tcgen05_acc_loaded")
        prelude_load = f"{prelude_indent}{loaded} = {load_expr}\n"
        chain_prelude, final_expr = epilogue_chain.render_prelude_and_expr(
            loaded, df.new_var, prelude_indent
        )
        return (prelude_load + chain_prelude, f"({final_expr}).to({target_dtype})")

    if tcgen05_value.use_tma_store_epilogue:
        df.placeholder_args.add(tensor_name)
        df.wrapper_only_params.extend(
            [tcgen05_value.tma_store_atom, tcgen05_value.tma_store_tensor]
        )
        if tcgen05_value.use_role_local_epi:
            df.register_cute_tcgen05_epi_role_tile_counter(
                tcgen05_value.role_local_tile_counter
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

    def store_common_setup(
        gmem_tensor: str, *, include_full_tile: bool
    ) -> tuple[list[str], list[str]]:
        static_setup = [
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
                # `layout_c=` / `elem_ty_c=` match the D-output dtype so the
                # helper picks the with-source branch; the matmul-plan
                # `tcgen05_epi_tile` and the wrapper-side TMA atom must use
                # the same call shape (see `_make_tcgen05_layout_plan_setup`).
                f"{epi_tile} = cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape("
                f"({tcgen05_bm}, {tcgen05_bn}), False, "
                f"cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {target_dtype}, "
                f"layout_c=cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                f"elem_ty_c={target_dtype})"
            ),
        ]
        tile_setup: list[str] = []
        if include_full_tile:
            tile_setup.append(
                f"{full_tile} = "
                f"({base_indices[0]}) + cutlass.Int32({tcgen05_bm}) <= {m_size} "
                f"and ({base_indices[1]}) + cutlass.Int32({tcgen05_bn}) <= {n_size}"
            )
        tile_setup.extend(
            [
                (
                    f"{gmem_tile} = cute.local_tile("
                    f"{gmem_tensor}, ({tcgen05_bm}, {tcgen05_bn}), "
                    f"({tile_coord_m}, {tile_coord_n}))"
                ),
                f"{tcgc_base} = {tcgen05_thr_mma}.partition_C({gmem_tile})",
            ]
        )
        return static_setup, tile_setup

    simt_static_store_setup, simt_tile_store_setup = store_common_setup(
        tensor_name, include_full_tile=True
    )
    simt_acc_vec_prelude, simt_acc_vec_rhs = _splice_acc_vec(ttr_racc, "        ")
    tma_static_store_setup, tma_tile_store_setup = store_common_setup(
        tcgen05_value.tma_store_tensor, include_full_tile=False
    )
    tma_c_buffer_expr = "cutlass.Int32(_tcgen05_subtile)"
    if tcgen05_value.role_local_tile_counter:
        tma_c_buffer_expr = (
            f"{tcgen05_value.role_local_tile_counter} * "
            f"cutlass.Int32({subtile_count}) + cutlass.Int32(_tcgen05_subtile)"
        )
    simt_store_body_core = [
        *simt_static_store_setup,
        *simt_tile_store_setup,
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
            f"{simt_acc_vec_prelude}"
            f"        {acc_vec} = {simt_acc_vec_rhs}\n"
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
            # Advance is a per-thread local state update, so it intentionally
            # stays outside elect_one; only the mbarrier release is elected.
            f"if {tcgen05_value.epi_active}:\n"
            + emit_pipeline_advance(tcgen05_value.acc_consumer_state, indent="    ")
        ),
    ]
    tma_store_pipeline_setup = [
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
    ]
    tma_store_pipeline_tail = (
        f"if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
        f"    {c_pipeline}.producer_tail()"
    )
    c_acquire_placement = state.device_function.config.get(
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    )
    acc_wait_placement = state.device_function.config.get(
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
        TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    )
    c_store_mode = state.device_function.config.get(
        TCGEN05_C_STORE_MODE_CONFIG_KEY,
        TCGEN05_C_STORE_MODE_NORMAL,
    )
    epilogue_layout = state.device_function.config.get(
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    )
    diagnose_first_c_acquire_in_loop = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
    )
    diagnose_later_c_acquire_before_barrier = (
        c_acquire_placement == TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER
    )
    diagnose_acc_wait_before_subtile_loop = (
        acc_wait_placement == TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
    )
    diagnose_skip_epilogue_store = (
        c_store_mode == TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
    )
    diagnose_split_first_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
    )
    diagnose_split_acc_t2r_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL
    )
    diagnose_module_helper_acc_t2r = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R
    )
    diagnose_module_helper_store_tail = (
        epilogue_layout == TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL
    )
    diagnose_split_epilogue_layout = (
        diagnose_split_first_t2r
        or diagnose_split_acc_t2r_store_tail
        or diagnose_module_helper_acc_t2r
        or diagnose_module_helper_store_tail
    )
    if diagnose_split_epilogue_layout:
        if not (
            tcgen05_value.use_role_local_epi and tcgen05_value.use_tma_store_epilogue
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} "
                "requires the "
                "role-local TMA-store tcgen05 epilogue",
            )
        if not tcgen05_value.is_two_cta:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} requires "
                "CtaGroup.TWO",
            )
        # Conservative proxy for the validated static-full CtaGroup.TWO
        # two-or-more-subtile envelope; the exact subtile count is only
        # available after the CUTLASS epilogue partitioning below.
        if tcgen05_value.bn < TCGEN05_TWO_CTA_BLOCK_N:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY}={epilogue_layout!r} is only "
                f"validated for CtaGroup.TWO block_n >= {TCGEN05_TWO_CTA_BLOCK_N}",
            )
    tma_store_first_subtile_acquire = (
        []
        if diagnose_first_c_acquire_in_loop
        else [
            (
                f"if {tcgen05_value.epi_active} and "
                f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
                f"    {c_pipeline}.producer_acquire()"
            )
        ]
    )
    tma_store_loop_first_subtile_acquire = (
        (
            f"        if _tcgen05_subtile == 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_first_c_acquire_in_loop
        else ""
    )
    tma_store_split_first_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_first_c_acquire_in_loop
        else ""
    )
    tma_store_loop_later_subtile_acquire = (
        ""
        if diagnose_later_c_acquire_before_barrier
        else (
            f"        if _tcgen05_subtile != 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
    )
    tma_store_loop_late_later_subtile_acquire = (
        (
            f"        if _tcgen05_subtile != 0 and "
            f"{tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_later_c_acquire_before_barrier
        else ""
    )
    tma_store_pre_loop_acc_wait = (
        [
            (
                f"if {tcgen05_value.epi_active}:\n"
                f"    {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})"
            )
        ]
        if diagnose_acc_wait_before_subtile_loop
        else []
    )
    tma_store_loop_acc_wait = (
        ""
        if diagnose_acc_wait_before_subtile_loop
        else (
            f"        if _tcgen05_subtile == 0:\n"
            f"            {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})\n"
        )
    )
    tma_store_split_first_acc_wait = (
        ""
        if diagnose_acc_wait_before_subtile_loop
        else (
            f"        {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})\n"
        )
    )
    tma_store_split_tail_later_subtile_acquire = (
        ""
        if diagnose_later_c_acquire_before_barrier
        else (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
    )
    tma_store_split_tail_late_later_subtile_acquire = (
        (
            f"        if {tcgen05_value.warp_idx} == cutlass.Int32(0):\n"
            f"            {c_pipeline}.producer_acquire()\n"
        )
        if diagnose_later_c_acquire_before_barrier
        else ""
    )
    # Pyrefly does not preserve the non-None tcgen05_value narrowing inside
    # the nested source formatter, so keep local string aliases for attributes
    # read only by that closure.
    tcgen05_epi_active = tcgen05_value.epi_active
    tcgen05_acc_pipeline = tcgen05_value.acc_pipeline
    tcgen05_acc_consumer_state = tcgen05_value.acc_consumer_state
    tcgen05_warp_idx = tcgen05_value.warp_idx
    tcgen05_tma_store_atom = tcgen05_value.tma_store_atom

    def tma_store_acc_t2r_region(*, acc_wait: str) -> str:
        prelude, rhs = _splice_acc_vec(trs_racc, "        ")
        return (
            f"{acc_wait}"
            f"        {ttr_tacc_mn} = {ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            f"        cute.copy({tiled_copy_t2r}, {ttr_tacc_mn}, {ttr_racc})\n"
            f"{prelude}"
            f"        {acc_vec} = {rhs}\n"
            f"        if _tcgen05_subtile == {subtile_count} - 1:\n"
            f"            cute.arch.fence_view_async_tmem_load()\n"
            f"            with cute.arch.elect_one():\n"
            f"                {tcgen05_acc_pipeline}.consumer_release({tcgen05_acc_consumer_state})\n"
            f"        {trs_rd}.store({acc_vec})\n"
        )

    def tma_store_tail_region(*, late_later_subtile_acquire: str) -> str:
        return (
            f"{late_later_subtile_acquire}"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        {c_buffer} = ({tma_c_buffer_expr}) % cutlass.Int32({tcgen05_c_stage_count})\n"
            f"        cute.copy({tiled_copy_r2s}, {trs_rd}, {trs_sd}[(None, None, None, {c_buffer})])\n"
            f"        cute.arch.fence_view_async_shared()\n"
            f"        {epilog_sync_barrier}.arrive_and_wait()\n"
            f"        if {tcgen05_warp_idx} == cutlass.Int32(0):\n"
            f"            cute.copy({tcgen05_tma_store_atom}, {bsg_sd}[(None, {c_buffer})], {bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            f"            {c_pipeline}.producer_commit()\n"
        )

    def tma_store_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_acc_t2r_region(acc_wait=acc_wait)}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire)}"
        )

    def indented_diagnostic_region(source: str) -> str:
        if not source:
            return "            pass\n"
        return "".join(f"    {line}" for line in source.splitlines(keepends=True))

    def tma_store_helper_boundary_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
        late_later_subtile_acquire: str,
    ) -> str:
        acquire_region = f"{first_subtile_acquire}{later_subtile_acquire}"
        acc_region = tma_store_acc_t2r_region(acc_wait=acc_wait)
        tail_region = tma_store_tail_region(
            late_later_subtile_acquire=late_later_subtile_acquire
        )
        # These constant-true blocks are diagnostic source boundaries. The
        # generated-code AST round trip preserves them, while emitted comments
        # are not reliable line-info anchors.
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"        if True:\n"
            f"{indented_diagnostic_region(acquire_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(acc_region)}"
            f"        if True:\n"
            f"{indented_diagnostic_region(tail_region)}"
        )

    module_acc_t2r_helper_name = (
        df.unique_name("tcgen05_acc_t2r_region")
        if diagnose_module_helper_acc_t2r
        else ""
    )
    module_store_tail_helper_name = (
        df.unique_name("tcgen05_store_tail_region")
        if diagnose_module_helper_store_tail
        else ""
    )

    def tma_store_module_acc_t2r_helper_source(*, acc_wait: str) -> str:
        prelude, rhs = _splice_acc_vec("tcgen05_tRS_rAcc", "    ")
        return (
            "@cute.jit\n"
            f"def {module_acc_t2r_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_acc_pipeline, "
            "tcgen05_acc_consumer_state, "
            "tcgen05_tTR_tAcc, "
            "tcgen05_tiled_copy_t2r, "
            "tcgen05_tTR_rAcc, "
            "tcgen05_tRS_rAcc, "
            "tcgen05_tRS_rD, "
            "tcgen05_subtile_count"
            "):\n"
            f"{acc_wait}"
            "    tcgen05_tTR_tAcc_mn = tcgen05_tTR_tAcc[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
            "    cute.copy(tcgen05_tiled_copy_t2r, tcgen05_tTR_tAcc_mn, tcgen05_tTR_rAcc)\n"
            f"{prelude}"
            f"    tcgen05_acc_vec = {rhs}\n"
            "    if _tcgen05_subtile == tcgen05_subtile_count - 1:\n"
            "        cute.arch.fence_view_async_tmem_load()\n"
            "        with cute.arch.elect_one():\n"
            "            tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)\n"
            "    tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        )

    def tma_store_module_acc_t2r_helper_call() -> str:
        return (
            f"        {module_acc_t2r_helper_name}("
            f"_tcgen05_subtile, "
            f"{tcgen05_acc_pipeline}, "
            f"{tcgen05_acc_consumer_state}, "
            f"{ttr_tacc}, "
            f"{tiled_copy_t2r}, "
            f"{ttr_racc}, "
            f"{trs_racc}, "
            f"{trs_rd}, "
            f"{subtile_count})\n"
        )

    def tma_store_module_helper_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        late_later_subtile_acquire: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_module_acc_t2r_helper_call()}"
            f"{tma_store_tail_region(late_later_subtile_acquire=late_later_subtile_acquire)}"
        )

    def tma_store_module_tail_helper_source(*, late_later_subtile_acquire: str) -> str:
        return (
            "@cute.jit\n"
            f"def {module_store_tail_helper_name}("
            "_tcgen05_subtile, "
            "tcgen05_tma_c_buffer_index, "
            "tcgen05_epilog_sync_barrier, "
            "tcgen05_tiled_copy_r2s, "
            "tcgen05_tRS_rD, "
            "tcgen05_tRS_sD, "
            "tcgen05_tma_store_atom, "
            "tcgen05_bSG_sD, "
            "tcgen05_bSG_gD, "
            "tcgen05_c_pipeline, "
            "tcgen05_warp_idx"
            "):\n"
            f"{late_later_subtile_acquire}"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            f"    tcgen05_c_buffer = tcgen05_tma_c_buffer_index % cutlass.Int32({tcgen05_c_stage_count})\n"
            "    cute.copy(tcgen05_tiled_copy_r2s, tcgen05_tRS_rD, tcgen05_tRS_sD[(None, None, None, tcgen05_c_buffer)])\n"
            "    cute.arch.fence_view_async_shared()\n"
            "    tcgen05_epilog_sync_barrier.arrive_and_wait()\n"
            "    if tcgen05_warp_idx == cutlass.Int32(0):\n"
            "        cute.copy(tcgen05_tma_store_atom, tcgen05_bSG_sD[(None, tcgen05_c_buffer)], tcgen05_bSG_gD[(None, cutlass.Int32(_tcgen05_subtile))])\n"
            "        tcgen05_c_pipeline.producer_commit()"
        )

    def tma_store_module_tail_helper_call() -> str:
        return (
            f"        {module_store_tail_helper_name}("
            f"_tcgen05_subtile, "
            f"{tma_c_buffer_expr}, "
            f"{epilog_sync_barrier}, "
            f"{tiled_copy_r2s}, "
            f"{trs_rd}, "
            f"{trs_sd}, "
            f"{tcgen05_tma_store_atom}, "
            f"{bsg_sd}, "
            f"{bsg_gd}, "
            f"{c_pipeline}, "
            f"{tcgen05_warp_idx})\n"
        )

    def tma_store_module_tail_subtile_body(
        *,
        first_subtile_acquire: str,
        later_subtile_acquire: str,
        acc_wait: str,
    ) -> str:
        return (
            f"    if {tcgen05_epi_active}:\n"
            f"{first_subtile_acquire}"
            f"{later_subtile_acquire}"
            f"{tma_store_acc_t2r_region(acc_wait=acc_wait)}"
            f"{tma_store_module_tail_helper_call()}"
        )

    if diagnose_split_first_t2r:
        tma_store_split_first_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_split_first_subtile_acquire,
            later_subtile_acquire="",
            acc_wait=tma_store_split_first_acc_wait,
            late_later_subtile_acquire="",
        )
        tma_store_split_tail_subtile_body = tma_store_subtile_body(
            first_subtile_acquire="",
            later_subtile_acquire=tma_store_split_tail_later_subtile_acquire,
            acc_wait="",
            late_later_subtile_acquire=(
                tma_store_split_tail_late_later_subtile_acquire
            ),
        )
        # Diagnostic-only scaffolding: reuse the one-indent subtile formatter
        # for a static first subtile without changing production source layout.
        # The tail loop maps split-loop indices back to logical subtile ids 1..N-1;
        # unroll_full=True keeps those subtile values compile-time constants.
        tma_store_subtile_loop = (
            "if True:\n"
            f"    _tcgen05_subtile = 0\n"
            f"{tma_store_split_first_subtile_body}"
            f"for _tcgen05_split_subtile in cutlass.range({subtile_count} - 1, unroll_full=True):\n"
            f"    _tcgen05_subtile = _tcgen05_split_subtile + 1\n"
            f"{tma_store_split_tail_subtile_body}"
        )
    elif diagnose_split_acc_t2r_store_tail:
        tma_store_helper_boundary_body = tma_store_helper_boundary_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_helper_boundary_body}"
        )
    elif diagnose_module_helper_acc_t2r:
        module_helper_acc_wait = (
            ""
            if diagnose_acc_wait_before_subtile_loop
            else (
                "    if _tcgen05_subtile == 0:\n"
                "        tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)\n"
            )
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_acc_t2r_helper_source(acc_wait=module_helper_acc_wait)
            )
        )
        tma_store_module_helper_body = tma_store_module_helper_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_helper_body}"
        )
    elif diagnose_module_helper_store_tail:
        module_tail_late_later_subtile_acquire = (
            (
                "    if _tcgen05_subtile != 0 and "
                "tcgen05_warp_idx == cutlass.Int32(0):\n"
                "        tcgen05_c_pipeline.producer_acquire()\n"
            )
            if diagnose_later_c_acquire_before_barrier
            else ""
        )
        state.codegen.module_statements.append(
            statement_from_string(
                tma_store_module_tail_helper_source(
                    late_later_subtile_acquire=module_tail_late_later_subtile_acquire
                )
            )
        )
        tma_store_module_tail_body = tma_store_module_tail_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_module_tail_body}"
        )
    else:
        tma_store_default_subtile_body = tma_store_subtile_body(
            first_subtile_acquire=tma_store_loop_first_subtile_acquire,
            later_subtile_acquire=tma_store_loop_later_subtile_acquire,
            acc_wait=tma_store_loop_acc_wait,
            late_later_subtile_acquire=tma_store_loop_late_later_subtile_acquire,
        )
        tma_store_subtile_loop = (
            f"for _tcgen05_subtile in cutlass.range({subtile_count}, unroll_full=True):\n"
            f"{tma_store_default_subtile_body}"
        )
    tma_store_smem_setup = [
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
    ]
    tma_store_acc_layout_setup = [
        (
            f"{tacc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgen05_value.epi_acc_frag_base})"
        ),
    ]
    tma_store_role_invariant_setup = [
        *tma_static_store_setup,
        *tma_store_smem_setup,
        *tma_store_acc_layout_setup,
    ]
    suppressed_store_body_core = [
        (
            # Diagnostic-only invalid-output mode. Keep the accumulator
            # pipeline draining so persistent kernels do not deadlock, but
            # suppress C-pipeline acquire/commit, R2S/SMEM work, and TMA D
            # stores to bound whether hot waits are tied to the C-store path.
            f"if {tcgen05_value.epi_active}:\n"
            f"    {tcgen05_value.acc_pipeline}.consumer_wait({tcgen05_value.acc_consumer_state})\n"
            f"    with cute.arch.elect_one():\n"
            f"        {tcgen05_value.acc_pipeline}.consumer_release({tcgen05_value.acc_consumer_state})\n"
            + emit_pipeline_advance(
                tcgen05_value.acc_consumer_state,
                indent="    ",
            )
        )
    ]
    # Non-role-local stores keep pipeline/SMEM setup before per-tile C
    # partitioning so the hoisted role-local prefix matches the same
    # invariant setup subset.
    tma_store_body_core = [
        *([] if tcgen05_value.use_role_local_epi else tma_static_store_setup),
        *([] if tcgen05_value.use_role_local_epi else tma_store_pipeline_setup),
        *([] if tcgen05_value.use_role_local_epi else tma_store_smem_setup),
        *tma_store_first_subtile_acquire,
        *tma_tile_store_setup,
        (
            f"{tcgc} = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            f"{tcgc_base})"
        ),
        (
            f"{tcgc_planned} = cute.make_tensor("
            f"{tcgc}.iterator, "
            f"cute.append(cute.append(cute.append({tcgc}.layout, {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}), {tcgen05_value.epilogue_rest_mode}))"
        ),
        *([] if tcgen05_value.use_role_local_epi else tma_store_acc_layout_setup),
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
            f"{ttr_tacc_stage} = {ttr_tacc_base}["
            f"(None, None, None, None, None, {tcgen05_value.acc_consumer_state}.index)]"
        ),
        f"{ttr_tacc} = cute.group_modes({ttr_tacc_stage}, 3, cute.rank({ttr_tacc_stage}))",
        f"{subtile_count} = cutlass.const_expr(cute.size({ttr_tacc}.shape, mode=[3]))",
        *tma_store_pre_loop_acc_wait,
        (
            # Warp 0 pre-acquires the first TMA-store SMEM stage before
            # per-tile C-store setup. The subtile loop acquires only later
            # stages, so C-stage waits can overlap setup, the first
            # acc-pipeline wait, and the other epi warps' TMEM
            # load/conversion work on later subtile iterations. The
            # diagnostic tcgen05_c_acquire_placement=first_in_loop moves only
            # that first acquire into the subtile loop; later acquires and
            # the accumulator wait keep their default order. The diagnostic
            # later_before_barrier placement keeps the first acquire in
            # production position and moves only later-subtile acquires just
            # before the first epilogue barrier. The diagnostic
            # tcgen05_acc_wait_placement=before_subtile_loop keeps both C
            # acquire sites in production position and moves only the
            # accumulator consumer wait before the subtile loop.
            # A CTA-scoped named barrier ensures all epi warps have observed
            # warp 0's acquire before they write SMEM; a second barrier ensures
            # the SMEM writes and Quack-style async-shared fence are visible
            # before warp 0 issues and commits the TMA operation.
            # Compute the SMEM ring index after the first barrier so the
            # acquire/barrier/index order stays aligned with Quack's
            # TMA-store epilogue.
            # The accumulator consumer state advances after the loop, matching
            # Quack's call-site ordering while preserving the early release.
            # After warp 0 commits the TMA store, the next subtile's
            # producer_acquire plus the first named barrier are enough to
            # keep all epi warps from writing a reused SMEM stage too early.
            # Avoiding a post-commit barrier matches Quack's epilogue loop.
            # The split_first_t2r diagnostic emits the first static subtile as
            # a standalone source block, then loops over later subtile work.
            # It is a layout discriminator for the hot acc-wait/T2R SASS row;
            # the default production source shape remains the single loop.
            tma_store_subtile_loop
            # Advance is a per-thread local state update, so it intentionally
            # stays outside elect_one; only the mbarrier release is elected.
            + f"if {tcgen05_value.epi_active}:\n"
            + emit_pipeline_advance(tcgen05_value.acc_consumer_state, indent="    ")
        ),
        *([] if tcgen05_value.use_role_local_epi else [tma_store_pipeline_tail]),
    ]
    store_body_core = (
        suppressed_store_body_core
        if diagnose_skip_epilogue_store
        else (
            tma_store_body_core
            if tcgen05_value.use_tma_store_epilogue
            else simt_store_body_core
        )
    )
    main_stmts: list[ast.AST]
    if tcgen05_value.use_role_local_epi:
        # These setup statements intentionally remain virtual-pid-independent.
        # The persistent splitter hoists them before the role-local scheduler
        # loops; if future setup reads per-tile state, it must be registered
        # as per-tile work instead.
        tma_store_hoisted_stmts = (
            [
                statement_from_string(line)
                for line in [
                    *tma_store_pipeline_setup,
                    *tma_store_role_invariant_setup,
                ]
            ]
            if tcgen05_value.use_tma_store_epilogue and not diagnose_skip_epilogue_store
            else []
        )
        sync_before_stmt = statement_from_string("cute.arch.sync_threads()")
        main_stmt = statement_from_string(
            "if True:\n" + textwrap.indent("\n".join(store_body_core), "    ")
        )
        sync_after_stmt = statement_from_string("cute.arch.sync_threads()")
        df.register_cute_tcgen05_per_tile_stmts(
            [sync_before_stmt, main_stmt, sync_after_stmt]
        )
        df.register_cute_tcgen05_epi_role_stmts([main_stmt])
        main_stmts = [
            *tma_store_hoisted_stmts,
            sync_before_stmt,
            main_stmt,
            sync_after_stmt,
        ]
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
    if (
        tcgen05_value.use_tma_store_epilogue
        and tcgen05_value.use_role_local_epi
        and not diagnose_skip_epilogue_store
    ):
        # Role-local persistent epilogues reuse the C-store pipeline across
        # scheduler-recycled work tiles. Draining it inside each tile would
        # serialize the next tile's epilogue against this tile's TMA stores.
        # The tail must run before TMEM dealloc setup below.
        post_loop_lines.append(tma_store_pipeline_tail)
    if tcgen05_value.use_tma:
        post_loop_lines.append(
            f"if {tcgen05_value.tma_warp}:\n"
            + emit_producer_tail_tma_umma(
                tcgen05_value.tma_pipeline,
                tcgen05_value.tma_producer_state,
                num_stages=tcgen05_value.ab_stage_count,
                indent="    ",
                skip_advances=tcgen05_value.skip_ab_producer_advance,
            )
        )
    if tcgen05_value.is_two_cta:
        # PDL parity with Quack/CUTLASS: after all MMAs are issued, hint
        # dependent kernels before this role starts the final acc drain.
        post_loop_lines.append(
            f"if {tcgen05_value.exec_active}:\n"
            "    cute.arch.griddepcontrol_launch_dependents()"
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
        ]
    )
    if not tcgen05_value.is_two_cta:
        # Keep the long-validated cluster_m=1 teardown unchanged. The guarded
        # CtaGroup.TWO path follows Quack's dealloc sequence without this CTA
        # sync: epi warps synchronize through tmem_alloc_barrier before free.
        post_loop_lines.append("cute.arch.sync_threads()")
    post_loop_lines.extend(
        [
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


def _try_splice_tcgen05_unary_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """G3.1-B splice attempt for ``out[tile] = unary_chain(acc).to(x.dtype)``.

    Returns the splice-completion sentinel (``ast.Constant(value=None)``)
    on a successful splice (the caller should return it directly), and
    ``None`` if the splice did not fire — the caller should continue to
    the G3.1.0 backstop or the SIMT fallback.

    Splice is attempted only when the kernel has a tcgen05-registered
    matmul fx_node (``cute_tcgen05_matmul_fx_nodes`` non-empty), the
    store value has a backing FX node, the store target is a 2-D
    ``torch.Tensor``, and the chain analyzer accepts the value chain
    (returning ``(chain, anchor)`` for a non-empty unary chain rooted
    at a tcgen05 matmul). Chains the whitelist rejects (auxiliary-
    tensor lambdas, reductions, kwarg-bearing scalar binaries, etc.)
    leave the analyzer returning ``None`` and the splice does not
    fire — the G3.1.0 backstop then catches them.
    """
    if not state.device_function.cute_tcgen05_matmul_fx_nodes:
        return None
    if value_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    analyzed = analyze_tcgen05_unary_epilogue_chain(state, value_node)
    if analyzed is None:
        return None
    chain, anchor = analyzed
    # Defensive: the analyzer contract is to return ``None`` for
    # identity stores (no chain steps), so a non-None return here
    # always carries a non-empty chain. Pin the invariant.
    assert chain.steps, (
        "analyze_tcgen05_unary_epilogue_chain must return None for "
        "identity stores (no chain steps); reaching the splice site "
        "with an empty chain indicates a contract violation"
    )
    anchor_result_var = (
        state.device_function.cute_tcgen05_matmul_fx_node_result_vars.get(anchor)
    )
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        epilogue_chain=chain,
    )
    if rewritten_stmt is None:
        return None
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


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
                rewritten_stmt = _codegen_cute_store_stack_load(
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
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        value = expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value,
        )
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        store_expr = expr_from_string(
            "({ptr}).store({value})", ptr=ptr_expr, value=value
        )
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if mask_expr is None:
            return store_expr
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(store_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=store_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
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

    # G3.1.1: try to splice a whitelisted unary-chain epilogue
    # (`out[tile] = unary_chain(acc).to(x.dtype)`) into the role-local
    # tcgen05 epilogue's per-thread T2R loop. Implementation in
    # ``_try_splice_tcgen05_unary_epilogue``. Chains the whitelist
    # rejects (auxiliary-tensor lambdas, reductions, etc.) leave the
    # splice off and fall through to the G3.1.0 backstop below.
    spliced = _try_splice_tcgen05_unary_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        return spliced

    # Loud-failure backstop for fused-epilogue stores that follow a
    # tcgen05 matmul. The tcgen05 grid-emission path (in `program_id.py`)
    # does not bind the per-block-id `indices_<n>` / `mask_<n>` variable
    # names that the SIMT-fallback store path expects, so falling through
    # here would emit a kernel that crashes inside the cute DSL with
    # `name 'mask_0' is not defined`. Detect the pattern here — any
    # store value whose FX user chain transitively reaches a
    # tcgen05-registered matmul fx node — and raise a structured error
    # so the caller sees the actionable message instead of a cute-DSL
    # crash. Fixing this requires either (a) extending the tcgen05 grid
    # to emit per-block-id index/mask vars, or (b) per-subtile lambda
    # emission in `_codegen_cute_store_tcgen05_tile`.
    if (
        state.device_function.cute_tcgen05_matmul_fx_nodes
        and value_node is not None
        and reach_tcgen05_matmul_anchors(state, value_node)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 MMA path does not yet emit per-block-id indices "
            "and masks for non-whitelisted or auxiliary-input fused "
            "epilogues that follow the MMA. The store target's value "
            "chain depends on a tcgen05 matmul result through ops the "
            "G3.1-B chain analyzer rejects (e.g. auxiliary tensor loads "
            "such as `acc + bias[tile_n]` or `acc + residual[tile]`, "
            "non-scalar binary ops, `aten.add.Tensor` with `alpha=k`, "
            "or an intermediate `.to(d_inter)` cast where `d_inter` "
            "differs from the store-target dtype). Identity stores "
            "(`out[tile] = acc.to(x.dtype)`) and whitelisted unary "
            "chains (relu/tanh/exp/log/sqrt/abs/neg + scalar "
            "add/sub/mul/div on the accumulator carrier) do work via "
            "the G3.1-B fused-unary splice path. Auxiliary-tensor "
            "fusion is queued as G3.1-C. See cute_plan.md §7.5.",
        )

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
    store_uses_pointer = "None" not in index_exprs
    store_expr = _cute_scalar_store_expr(tensor_name, index_exprs, "{value}")
    assign_expr = expr_from_string(store_expr, value=value)

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        topk_mask = f"({topk_lane_expr}) < {topk_k}"
        mask_expr = topk_mask if mask_expr is None else f"({mask_expr}) and {topk_mask}"
    if mask_expr is None:
        return assign_expr
    if store_uses_pointer:
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(assign_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=assign_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    return expr_from_string(
        f"({store_expr} if {mask_expr} else None)",
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
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        load_expr = f"({ast.unparse(ptr_expr)}).load()"
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if tensor_like.dtype is torch.bool:
            load_expr = f"({load_expr} != cutlass.Uint8(0))"
            if mask_expr is None:
                return expr_from_string(load_expr)
            return expr_from_string(
                f"({load_expr} if {mask_expr} else cutlass.Boolean(0))"
            )
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else {target_dtype}(0))")
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
    load_expr = _cute_scalar_load_expr(tensor_name, index_exprs)
    mask_expr = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor,
        include_tensor_index_masks=False,
    )
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
