"""CuTe-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import map_arg

from ... import exc
from ...language import _decorators
from ...language.memory_ops import _CUTE_VECTOR_DTYPES
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_DTYPES
from ...language.memory_ops import _codegen_cute_store_permute_lane_loops
from ...language.memory_ops import _codegen_cute_store_tcgen05_tile
from ...language.memory_ops import _cute_active_index_var
from ...language.memory_ops import _cute_active_mask_var
from ...language.memory_ops import _cute_combined_mask
from ...language.memory_ops import _cute_index_exprs
from ...language.memory_ops import _cute_index_tuple
from ...language.memory_ops import _cute_is_byte_packed
from ...language.memory_ops import _cute_is_unroll_dtype
from ...language.memory_ops import _cute_register_tile_unroll_vec_hoist
from ...language.memory_ops import _cute_register_tile_unroll_vec_hoist_split2
from ...language.memory_ops import _cute_scalar_load_expr
from ...language.memory_ops import _cute_scalar_pointer_expr
from ...language.memory_ops import _cute_tensor_dim_size_expr
from ...language.memory_ops import _cute_unique_graph_block_id
from ...language.memory_ops import _cute_unroll_vec_elem_type
from ...language.memory_ops import _matching_block_ids
from ...language.memory_ops import _maybe_codegen_cute_packed_affine_lhs_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .cute_fx_walk import reach_tcgen05_matmul_anchors

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

log = logging.getLogger(__name__)


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


def _maybe_codegen_cute_packed_rhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    from .indexing import match_cute_duplicate_stack_reshape_rhs

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


def _cute_scalar_storage_dtype(dtype: torch.dtype) -> str:
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return "cutlass.Uint8"
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _cute_scalar_store_expr(
    tensor_name: str, index_exprs: list[str], value: str
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}.__setitem__({_cute_index_tuple(index_exprs)}, {value})"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.store({value})"


def _cute_unroll_vec_load_dtype_arg(dtype: torch.dtype, vec_width: int) -> str:
    """The dtype argument to ``cute.arch.load`` for an unroll-mode hoist.

    fp8 loads ``vec_width`` contiguous bytes as ONE packed scalar integer
    (no ``VectorType`` — avoids the V=8 ``nvvm.load.ext`` ICE and emits a
    single LDG).  bf16/fp16 load a ``Uint16`` vector of width ``vec_width``.
    """
    if _cute_is_byte_packed(dtype):
        return _cute_unroll_vec_elem_type(dtype, vec_width) + ".mlir_type"
    return f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type)"


def _cute_vector_load_expr(
    tensor_name: str,
    index_exprs: list[str],
    dtype: torch.dtype,
    *,
    vec_width: int,
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    return (
        f"cute.arch.load({ptr}, ir.VectorType.get([{vec_width}], {elem_str}.mlir_type))"
    )


def _cute_vector_store_expr(
    tensor_name: str,
    index_exprs: list[str],
    value: str,
    dtype: torch.dtype,
    *,
    vec_width: int,
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    return (
        f"cute.arch.store({ptr}, {value}, "
        f"ir.VectorType.get([{vec_width}], {elem_str}.mlir_type))"
    )


def _cute_register_unroll_vec_hoist(
    state: CodegenState,
    strategy: object,  # LoopedReductionStrategy at runtime
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
) -> str:
    """Register a Uint16 vec load to be hoisted above the constexpr V-loop
    in the active lane body and return the per-element extract expression.

    The hoist runs once per outer-lane iter; the constexpr V-loop's body
    receives ``hoist_var[vi].bitcast(dtype)`` (a scalar) so the existing
    cast/mul/accumulate pipeline keeps working unchanged.
    """
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    base_index_var = getattr(strategy, "_cute_lane_base_index_var", None)
    lane_body = getattr(strategy, "_cute_lane_body", None)
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    # The inner reduction-axis index_expr is the last entry; swap it with
    # the per-lane base so the vec load points at the start of the V-wide
    # chunk this thread owns.
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    cache_key = (tensor_name, base_ptr_expr)
    cache = getattr(strategy, "_cute_lane_vec_loads", None)
    if cache is None:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads = cache
    if cache_key not in cache:
        hoist_var = state.device_function.new_var(
            f"_unroll_vec_{len(cache)}", dce=False
        )
        cache[cache_key] = (hoist_var, tensor.dtype)
        hoist_stmt = statement_from_string(
            f"{hoist_var} = cute.arch.load({base_ptr_expr}, "
            f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type))"
        )
        # Insert the hoist just BEFORE the constexpr V-loop (the last entry
        # in lane_body).  ``lane_body[-1]`` is the constexpr loop.
        lane_body.insert(len(lane_body) - 1, hoist_stmt)
    else:
        hoist_var, _ = cache[cache_key]
    # The constexpr V-loop's target var is the last element's loop var.
    constexpr_loop = lane_body[-1]
    assert isinstance(constexpr_loop, ast.For)
    assert isinstance(constexpr_loop.target, ast.Name)
    vec_lane_var = constexpr_loop.target.id
    return f"cutlass.Uint16({hoist_var}[{vec_lane_var}]).bitcast({elem_dtype})"


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
    from .indexing import CuteAffineRangeIndex

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
    from .indexing import CuteAffineRangeIndex

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
    from ..ast_extension import create
    from .indexing import CuteAffineRangeIndex

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


def _codegen_cute_affine_reshape_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Lower a 2-D affine-row store fed by a reshape/stack chain.

    Handles ``out[(begin*K):(begin*K + block*K), tile_n] = reshaped`` where the
    leading index is a ``CuteAffineRangeIndex`` (factor ``K``) over the m-tile,
    the trailing index is the n-tile, and the value is a row-major shape chain
    (e.g. ``stack([a, b], dim=1).reshape(block*K, block_n)``).

    Each m-tile thread owns row ``m_local`` of the source; the reshaped tensor
    has ``K`` rows per source row, so the thread loops ``s in range(K)`` and
    writes the value resolved at flat index ``(K*m_local + s)*block_n + n_local``
    to output row ``K*m_global + s``, column ``n_global``.
    """
    from ..ast_extension import create
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_block_local_coord
    from .cute_reshape import resolve_cute_shape_chain_value_at
    from .indexing import CuteAffineRangeIndex
    from .indexing import is_cute_shape_chain_target

    if (
        tensor.ndim != 2
        or len(subscript) != 2
        or len(ast_subscript) != 2
        or extra_mask is not None
        or value_node is None
        or not isinstance(state.codegen, GenerateAST)
    ):
        return None
    affine = ast_subscript[0]
    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    n_index = subscript[1]
    if not isinstance(n_index, torch.SymInt):
        return None
    env = CompileEnvironment.current()
    block_id_n = env.get_block_id(n_index)
    if block_id_n is None:
        return None
    block_id_m = _cute_affine_range_block_id(state, affine)
    if block_id_m is None:
        return None

    if value_node.op != "call_function" or not is_cute_shape_chain_target(
        value_node.target
    ):
        return None
    value_val = value_node.meta.get("val")
    if not isinstance(value_val, torch.Tensor) or value_val.ndim != 2:
        return None

    m_global = _cute_active_index_var(state, block_id_m)
    n_global = _cute_active_index_var(state, block_id_n)
    if m_global is None or n_global is None:
        return None
    m_local = _get_block_local_coord(state.codegen, block_id_m)
    n_local = _get_block_local_coord(state.codegen, block_id_n)
    if m_local is None or n_local is None:
        return None
    block_n = state.device_function.resolved_block_size(block_id_n)
    if not isinstance(block_n, int):
        return None

    factor = affine.factor
    lane_var = state.device_function.new_var("affine_lane", dce=True)
    row_local = f"cutlass.Int32({factor}) * ({m_local}) + cutlass.Int32({lane_var})"
    flat_index = (
        f"(({row_local}) * cutlass.Int32({block_n})) + ({n_local})"
        if block_n != 1
        else f"({row_local}) + ({n_local})"
    )
    value_ast = resolve_cute_shape_chain_value_at(state, value_node, flat_index)
    if value_ast is None:
        return None

    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(ast.unparse(value_ast), target_dtype)

    # Bind the resolved (possibly select-based) value to a variable so the CuTe
    # DSL sees the stack `ifexp` as its own assignment rather than nested inside
    # the `.store(...)` call / masked store ternary.
    value_var = state.device_function.new_var("affine_value", dce=True)

    row_index = (
        f"{index_dtype}(cutlass.Int32({factor}) * ({m_global}) "
        f"+ cutlass.Int32({lane_var}))"
    )
    col_index = f"{index_dtype}({n_global})"
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = _cute_scalar_store_expr(tensor_name, [row_index, col_index], value_var)

    store_stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    mask_parts = [
        mask
        for mask in (
            _cute_active_mask_var(state, block_id_m),
            _cute_active_mask_var(state, block_id_n),
        )
        if mask is not None
    ]
    if mask_parts:
        # Use a guard statement (not a ternary) so the CuTe DSL accepts the
        # device-value mask condition.
        mask_ast = expr_from_string(" and ".join(mask_parts))
        assert isinstance(mask_ast, ast.expr)
        store_stmt = ast.fix_missing_locations(
            ast.If(test=mask_ast, body=[store_stmt], orelse=[])
        )

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({factor})"),
        body=[
            statement_from_string(f"{value_var} = {value_expr}"),
            store_stmt,
        ],
        orelse=[],
        type_comment=None,
    )


def _is_cute_affine_range_load_for_store(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
) -> bool:
    from .indexing import CuteAffineRangeIndex
    from .indexing import match_cute_affine_range_iota

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
    from ..ast_extension import create

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


def _codegen_cute_store_loaded_index_trailing_slices(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from ..ast_extension import create

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
        block_size = state.device_function.resolved_block_size(block_id)
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


def _cute_expand_broadcast_dim(value_node: torch.fx.Node) -> int | None:
    """Return the dim an ``aten.expand`` broadcasts (input size 1 -> >1).

    Returns ``None`` unless ``value_node`` is an ``aten.expand`` whose value has
    exactly one broadcast dimension — i.e. the expanded value carries a stride-0
    mode at exactly one position whose pre-expand extent was 1. This is the
    signal that the stored value replicates one source element across that dim.
    """
    if value_node.target is not torch.ops.aten.expand.default:
        return None
    input_arg = value_node.args[0]
    if not isinstance(input_arg, torch.fx.Node):
        return None
    out_val = value_node.meta.get("val")
    in_val = input_arg.meta.get("val")
    if not isinstance(out_val, torch.Tensor) or not isinstance(in_val, torch.Tensor):
        return None
    if out_val.ndim != in_val.ndim:
        return None
    env = CompileEnvironment.current()
    broadcast_dims = [
        dim
        for dim in range(out_val.ndim)
        if env.known_equal(in_val.shape[dim], 1)
        and not env.known_equal(out_val.shape[dim], 1)
        and out_val.stride(dim) == 0
    ]
    if len(broadcast_dims) != 1:
        return None
    return broadcast_dims[0]


def _cute_block_tile_begin_expr(state: CodegenState, block_id: int) -> str | None:
    """Return the *per-block* tile start for a tile mapped onto a thread axis.

    In the CuTe SIMT model a tile dimension is spread across a thread axis, so
    the strategy's ``index_var`` is the per-*thread* global index
    (``pid * block + thread_idx[axis]``). Subtracting the thread-local coordinate
    yields the per-*block* tile base (``pid * block``), shared by every thread in
    the tile — the correct anchor for a broadcast lane loop. Returns ``None`` when
    the block id has no active thread axis in this scope.
    """
    from .cute_reshape import _grid_local_coord_expr

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    loop_state = loops[-1]
    thread_axis = loop_state.block_thread_axes.get(block_id)
    global_index = loop_state.strategy.index_var(block_id)
    if thread_axis is None or global_index is None:
        return None
    local_coord = _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    return state.codegen.lift(
        expr_from_string(f"({global_index}) - ({local_coord})"),
        dce=True,
        prefix="tile_begin",
    ).id


def _cute_unsqueeze_expand_load_source(
    value_node: torch.fx.Node, broadcast_dim: int
) -> torch.fx.Node | None:
    """Return the ``hl.load`` feeding ``expand(val[..., None, ...])``.

    Walks ``value_node`` (an ``aten.expand``) back through a single
    unsqueeze-style subscript op (``val[:, None, :]`` inserting the broadcast dim)
    to the originating ``hl.load``. Returns ``None`` unless the chain is exactly
    that shape, so the caller falls back to the load-agnostic path.
    """
    from ...language.view_ops import subscript as subscript_op

    inner = value_node.args[0]
    if not isinstance(inner, torch.fx.Node):
        return None
    if inner.op == "call_function" and inner.target is subscript_op:
        index_arg = inner.args[1] if len(inner.args) > 1 else None
        if not isinstance(index_arg, (list, tuple)):
            return None
        # Exactly one ``None`` (the inserted broadcast dim) at ``broadcast_dim``.
        none_positions = [pos for pos, entry in enumerate(index_arg) if entry is None]
        if none_positions != [broadcast_dim]:
            return None
        load_node = inner.args[0]
    else:
        load_node = inner
    if (
        isinstance(load_node, torch.fx.Node)
        and load_node.op == "call_function"
        and load_node.target is load
        and len(load_node.args) >= 2
    ):
        return load_node
    return None


def _codegen_cute_store_expand_broadcast_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    """Lower a store whose value is broadcast across a reused tile dimension.

    Handles the pattern::

        val = hl.load(src, [tile, hl.arange(k)])  # (block, k)
        val_3d = val[:, None, :].expand(block, block, k)  # stride-0 middle dim
        hl.store(out, [idx[tile], tile.index, hl.arange(k)], val_3d)

    Here ``tile`` appears twice in the store index — once as a tensor indexer
    (``idx[tile]``) and once as the bare tile index (``tile.index``) — while the
    value is broadcast (stride 0) along the second (``tile.index``) position. The
    generic SIMT store lowers both positions onto ``tile``'s single thread axis,
    so each thread only writes the ``a == b`` diagonal of the ``(block, block)``
    block. Instead emit a sequential lane loop over the broadcast position so a
    thread holding ``val[a]`` writes the full ``out[idx[a], begin+b, :]`` row for
    every ``b`` in the tile, filling the block. ``val`` is broadcast, so every
    lane reads the same per-thread register.

    Returns ``None`` (a strict no-op) unless every gate matches, so existing
    kernels are byte-for-byte unchanged.
    """
    env = CompileEnvironment.current()
    broadcast_dim = _cute_expand_broadcast_dim(value_node)
    if broadcast_dim is None:
        return None
    if broadcast_dim >= len(subscript):
        return None
    broadcast_idx = subscript[broadcast_dim]
    # The broadcast position must be a bare tile index (a SymInt block id), and
    # that same block id must be reused by another (tensor) index position — the
    # collision the generic path mis-handles.
    if not isinstance(broadcast_idx, torch.SymInt):
        return None
    broadcast_block_id = env.get_block_id(broadcast_idx)
    if broadcast_block_id is None:
        return None
    block_size = state.device_function.resolved_block_size(broadcast_block_id)
    if not isinstance(block_size, int) or block_size <= 1:
        return None
    reused = False
    for pos, idx in enumerate(subscript):
        if pos == broadcast_dim:
            continue
        if isinstance(idx, torch.Tensor):
            for dim_size in idx.shape:
                if broadcast_block_id in _matching_block_ids(env, dim_size):
                    reused = True
                    break
        if reused:
            break
    if not reused:
        return None

    # Walk the value chain ``expand -> unsqueeze(None) -> load`` to recover the
    # source load. The stored value is a per-thread register holding ``val[a, c]``
    # whose coordinates live on the *load*'s thread axes; the store's own free
    # ``hl.arange`` index entries are distinct nodes that the synthetic-axis
    # machinery assigns to *different* axes. Reusing the load's coordinate for
    # those non-broadcast positions keeps the register and the store address on
    # the same thread axis (otherwise thread ``(a, c_load, c_store)`` would write
    # ``out[..., c_store] = val[a, c_load]`` for ``c_load != c_store``).
    load_node = _cute_unsqueeze_expand_load_source(value_node, broadcast_dim)
    load_coords: list[str] | None = None
    load_subscript_proxy: tuple[object, ...] | None = None
    if load_node is not None:
        load_tensor_node = load_node.args[0]
        load_subscript = load_node.args[1]
        if isinstance(load_tensor_node, torch.fx.Node) and isinstance(
            load_subscript, (list, tuple)
        ):
            load_tensor = load_tensor_node.meta.get("val")
            if isinstance(load_tensor, torch.Tensor):
                load_subscript_proxy = tuple(
                    map_arg([*load_subscript], lambda arg: arg.meta["val"])
                )
                load_subscript_ast = map_arg(
                    [*load_subscript], lambda arg: state.env[arg]
                )
                load_coords = _cute_index_exprs(
                    state,
                    [*load_subscript_proxy],
                    [*load_subscript_ast],
                    tensor=load_tensor,
                    inactive_singleton_slice_expr="0",
                )
                if len(load_coords) != load_tensor.ndim:
                    load_coords = None
                    load_subscript_proxy = None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != tensor.ndim or "None" in index_exprs:
        return None

    # Re-align each non-broadcast free-``hl.arange`` store position onto the
    # load's matching coordinate. Value dim ``d`` maps to load dim ``d`` before
    # the unsqueezed broadcast dim and ``d - 1`` after it. Only positions where
    # *both* the store and the matching load entry are free ``hl.arange`` index
    # tensors are remapped — a tensor *indexer* (``idx[tile]``) keeps its own
    # coordinate.
    if load_coords is not None and load_subscript_proxy is not None:
        for pos, idx in enumerate(subscript):
            if pos == broadcast_dim or not isinstance(idx, torch.Tensor):
                continue
            load_dim = pos if pos < broadcast_dim else pos - 1
            if not (0 <= load_dim < len(load_coords)):
                continue
            if isinstance(load_subscript_proxy[load_dim], torch.Tensor):
                index_exprs[pos] = load_coords[load_dim]

    # Replace the broadcast position's coordinate (currently the reused tile's
    # per-thread global index) with ``block_begin + lane`` so the lane loop sweeps
    # the full tile block, identically for every thread in the tile. ``block_begin``
    # is the *per-block* tile start (``global_index - local_coord``); in the CuTe
    # SIMT model the tile is mapped onto a thread axis, so the bare offset var
    # still carries the per-thread ``thread_idx`` lane and must be stripped.
    block_begin = _cute_block_tile_begin_expr(state, broadcast_block_id)
    if block_begin is None:
        return None
    lane_var = state.device_function.new_var("bcast_lane", dce=True)
    index_dtype = env.index_type()
    broadcast_coord = f"({block_begin}) + {index_dtype}({lane_var})"
    index_exprs[broadcast_dim] = broadcast_coord

    backend = env.backend
    target_dtype = backend.dtype_str(tensor.dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, index_exprs, "{value}"),
        value=value,
    )

    # Base mask excludes the broadcast position (its bound is enforced by the lane
    # bound below); other positions keep their tile/tensor masks.
    base_subscript = [
        slice(None) if pos == broadcast_dim else idx
        for pos, idx in enumerate(subscript)
    ]
    mask_expr = _cute_combined_mask(state, base_subscript, extra_mask, tensor=tensor)
    dim_size = _cute_tensor_dim_size_expr(state, tensor, broadcast_dim)
    lane_bound = f"({broadcast_coord}) < {dim_size}"
    mask_expr = lane_bound if mask_expr is None else f"({mask_expr}) and {lane_bound}"

    from ..ast_extension import create

    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    body_stmt: ast.stmt = ast.fix_missing_locations(
        ast.If(
            test=mask_ast,
            body=[ast.Expr(value=store_expr)],
            orelse=[],
        )
    )
    loop_stmt = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({block_size})"),
        body=[body_stmt],
        orelse=[],
        type_comment=None,
    )
    state.add_statement(loop_stmt)
    return ast.Constant(value=None)


def _try_splice_tcgen05_unary_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice attempt for ``out[tile] = chain(acc).to(x.dtype)``.

    Returns the splice-completion sentinel (``ast.Constant(value=None)``)
    on a successful splice (the caller should return it directly), and
    ``None`` if the splice did not fire — the caller should continue to
    the loud-failure backstop or the SIMT fallback.

    Splice is attempted only when the kernel has a tcgen05-registered
    matmul fx_node (``cute_state.matmul_fx_nodes`` non-empty), the
    store value has a backing FX node, the store target is a 2-D
    ``torch.Tensor``, and the chain analyzer accepts the value chain
    (returning ``(chain, anchor)`` for a non-empty chain rooted at
    a tcgen05 matmul). Chains the whitelist rejects (broadcast aux
    loads, reductions, kwarg-bearing binaries, etc.) leave the
    analyzer returning ``None`` and the splice does not fire — the
    loud-failure backstop then catches them.
    """
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    analyzed = analyze_tcgen05_unary_epilogue_chain(
        state, value_node, output_global_shape=tuple(tensor.shape)
    )
    if analyzed is None:
        return None
    chain, anchor = analyzed
    if not chain.steps:
        return None
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(anchor)
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
        affine_reshape_store = _codegen_cute_affine_reshape_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value_node,
        )
        if affine_reshape_store is not None:
            state.add_statement(affine_reshape_store)
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
                rewritten_stmt = _codegen_cute_store_expand_broadcast_tile(
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
            from .cute_reshape import codegen_cute_store_permute

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

    # Try to splice a whitelisted chain epilogue
    # (`out[tile] = chain(acc).to(x.dtype)`) into the role-local
    # tcgen05 epilogue's per-thread T2R loop. Implementation in
    # ``_try_splice_tcgen05_unary_epilogue``. Chains the whitelist
    # rejects (broadcast aux loads, reductions, etc.) leave the
    # splice off and fall through to the loud-failure backstop
    # below.
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
        state.device_function.cute_state.matmul_fx_nodes
        and value_node is not None
        and reach_tcgen05_matmul_anchors(state, value_node)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 MMA path does not yet emit per-block-id indices "
            "and masks for non-whitelisted fused epilogues that follow "
            "the MMA. The store target's value chain depends on a "
            "tcgen05 matmul result through ops the chain analyzer "
            "rejects (e.g. aux tensors with a 3-D underlying shape "
            "and a static collapse like `aux3d[tile_m, tile_n, 0]`, "
            "loads whose index expression is not exactly the "
            "carrier tile-id symbol, non-scalar binary ops, "
            "`aten.add.Tensor` with `alpha=k`, or an intermediate "
            "`.to(d_inter)` cast where `d_inter` differs from the "
            "store-target dtype). Identity stores "
            "(`out[tile] = acc.to(x.dtype)`), whitelisted unary chains "
            "(relu/tanh/exp/log/sqrt/abs/neg + scalar add/sub/mul/div "
            "on the accumulator carrier), exact-shape 2-D "
            "auxiliary-tensor binary ops (`acc + residual[tile_m, "
            "tile_n]`), and rank-1 trailing-axis (rowvec) broadcast "
            "aux loads (`acc + bias[tile_n]`) all work via the "
            "fused-epilogue splice path. The leading-axis rank-1 "
            "form (`acc + bias[tile_m]`) is rejected because a bare "
            "rank-1 RHS aligns to the trailing axis under PyTorch "
            "broadcasting; an explicit colvec broadcast must be "
            "written with `bias[tile_m][:, None]` / "
            "`.unsqueeze(-1)`.",
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


def _cute_load_feeds_sort_or_scan(load_node: object) -> bool:
    """Return True if ``load_node`` feeds a sort/topk/_associative_scan.

    Direct users (sort/topk and the scalar ``_associative_scan`` path) are
    matched immediately.  For a tuple ``_associative_scan`` the index stream is
    typically a ``load`` that flows through a chain of dtype-cast / shape ops
    (e.g. ``indices[tile].float().unsqueeze(1).expand_as(vals)``) before
    reaching the scan.  To recover a scalar load for that stream we follow the
    forward chain through those pass-through ops.
    """
    from torch.fx.node import Node

    from .indexing import is_cute_shape_chain_target

    if not isinstance(load_node, Node):
        return False

    passthrough_targets = (torch.ops.prims.convert_element_type.default,)
    seen: set[Node] = set()
    stack: list[Node] = [load_node]
    while stack:
        node = stack.pop()
        for user in node.users:
            if not isinstance(user, Node):
                continue
            target = user.target
            if (
                target in (torch.ops.aten.sort.default, torch.ops.aten.topk.default)
                or getattr(target, "__name__", None) == "_associative_scan"
            ):
                return True
            if (
                is_cute_shape_chain_target(target) or target in passthrough_targets
            ) and user not in seen:
                seen.add(user)
                stack.append(user)
    return False


def _cute_vector_load_ctx(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
    extra_mask: ast.AST | None,
) -> tuple[int, int, str] | None:
    """Return (vec_width, lane_block_id, mode) when a vec load may be emitted.

    ``mode`` is one of ``"vec"`` (explicit ``cute.arch.load(..., V)``) or
    ``"unroll"`` (per-element scalar bitcast inside a constexpr V-loop).
    Returns None when any predicate for a 128-bit gmem load fails, in which
    case the caller falls back to ``_cute_scalar_load_expr``.
    """
    from ..reduction_strategy import LoopedReductionStrategy

    env = CompileEnvironment.current()
    if env.backend.name != "cute":
        return None
    if extra_mask is not None:
        return None
    if "None" in index_exprs:
        return None
    if tensor.dtype not in _CUTE_VECTOR_DTYPES and not _cute_is_unroll_dtype(
        tensor.dtype
    ):
        return None
    # Only enable the vec path when the load's result eventually feeds a
    # reduction op.  The consume-sweep mixes the loaded vector with scalar
    # values (e.g. the post-reduction inverse-RMS), and broadcasting
    # scalar->vec is not supported by the CuTe DSL today.  When the load's
    # immediate user is a dtype cast (``to(torch.float32)``), the
    # ``"unroll"`` mode further down keeps the strategy on a per-element
    # scalar pipeline and the explicit-vec path is skipped — the explicit
    # ``cute.arch.load(ptr, ir.VectorType.get([V], dtype.mlir_type))`` form
    # would otherwise crash inside the CuTe DSL when subscripting bf16/fp16
    # vectors.
    fx_node = state.fx_node
    if fx_node is None:
        return None
    visited: set[torch.fx.Node] = set()
    pending = list(fx_node.users.keys())
    feeds_reduction = False
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        target_name = getattr(user.target, "__name__", "") or ""
        target_qualname = getattr(user.target, "_qualname", "") or ""
        if (
            "reduction" in target_name
            or "_inductor_lowering_extra" in target_name
            or "reduction" in target_qualname
        ):
            feeds_reduction = True
            break
        pending.extend(user.users.keys())
    # Note: ``feeds_reduction`` is required ONLY for the ``vec`` mode below;
    # the ``unroll`` mode also applies to the consume sweep where the load
    # result feeds an elementwise pipeline (no reduction).
    # The lane/vec axis must be a tensor dim that is stride-1 so that
    # consecutive lane iters fetch consecutive bytes.  For a row-major lhs
    # the reduction axis is the LAST subscript position; for a column-major
    # rhs (e.g. the K-major ``y`` of a tcgen05 fp8 matmul) it is the FIRST.
    # ``_cute_lane_axis_pos`` records the index_exprs position of that
    # stride-1 lane axis so the hoist substitutes the per-lane base there
    # (not blindly at ``[-1]``).
    # Find the stride-1 dim WITHOUT forcing specialization of a symbolic
    # stride: a contiguous dim has a concrete ``int`` stride of 1, so only
    # accept plain ints here.  Calling ``int()`` on a ``SymInt`` stride would
    # bake the (otherwise-dynamic) size into the kernel — see the
    # ``test_mark_static`` regression where ``int(stride(0))`` specialized
    # ``n``.
    stride1_tensor_dim: int | None = None
    for d in range(tensor.ndim):
        s = tensor.stride(d)
        if isinstance(s, int) and s == 1:
            stride1_tensor_dim = d
            break
    if stride1_tensor_dim is None:
        return None
    # Locate the non-None subscript carrying an active lane block.  Slices
    # resolve to the matching tensor-dim block via the strategy that's
    # currently active for that block.  Prefer the block sitting on the
    # stride-1 tensor dim (the true lane axis), and record its index_exprs
    # position.
    inner_block_id: int | None = None
    lane_axis_pos: int | None = None
    expr_pos = -1
    tensor_dim = 0
    for idx in subscript:
        if idx is None:
            continue
        expr_pos += 1
        if isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None and state.codegen.active_device_loops.get(bid):
                if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                    inner_block_id = bid
                    lane_axis_pos = expr_pos
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor_dim < tensor.ndim:
                dim_size = tensor.shape[tensor_dim]
                for cand_bid, bs in enumerate(env.block_sizes):
                    if not isinstance(bs.size, (int, torch.SymInt)):
                        continue
                    bs_numel = bs.numel
                    # Try a few candidate forms for the size equality
                    # check: sympy.Integer (most common via specialize()),
                    # int, and torch.SymInt all flow through known_equal
                    # after we coerce to plain int when possible.
                    bs_int: int | torch.SymInt | None
                    if isinstance(bs_numel, (int, torch.SymInt)):
                        bs_int = bs_numel
                    else:
                        try:
                            bs_int = int(bs_numel)
                        except (TypeError, ValueError):
                            bs_int = None
                    if bs_int is None:
                        continue
                    dim_int: int | torch.SymInt | None
                    if isinstance(dim_size, (int, torch.SymInt)):
                        dim_int = dim_size
                    else:
                        try:
                            dim_int = int(dim_size)
                        except (TypeError, ValueError):
                            dim_int = None
                    if dim_int is None:
                        continue
                    if env.known_equal(
                        bs_int, dim_int
                    ) and state.codegen.active_device_loops.get(cand_bid):
                        if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                            inner_block_id = cand_bid
                            lane_axis_pos = expr_pos
                        break
        tensor_dim += 1
    if inner_block_id is None or lane_axis_pos is None:
        return None
    loops = state.codegen.active_device_loops.get(inner_block_id)
    if not loops:
        return None
    strategy = getattr(loops[-1], "strategy", None)
    if isinstance(strategy, LoopedReductionStrategy):
        vec_width = getattr(strategy, "_cute_reduction_vec_width", 1)
        if vec_width <= 1:
            return None
        if strategy._mask_var is not None:
            return None
        if strategy._cute_reduction_lane_extent <= 0:
            return None
        mode = getattr(strategy, "_cute_reduction_vec_mode", "vec")
        if mode == "vec":
            if not feeds_reduction:
                return None
            if tensor.dtype not in _CUTE_VECTOR_DTYPES:
                return None
            return vec_width, inner_block_id, "vec"
        if mode == "unroll":
            if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
                return None
            # The CuTe DSL's ``nvvm.load.ext`` only supports vec sizes 2
            # and 4 for bf16/fp16 (V=8 raises ICE).  Cap effective V
            # here so the autotuner's V=8 seed still compiles instead
            # of crashing.
            if vec_width > 4:
                return None
            # Need a lane base index var + a constexpr V-loop var; both
            # are set up by the strategy's codegen_device_loop.
            if (
                getattr(strategy, "_cute_lane_base_index_var", None) is None
                or getattr(strategy, "_cute_lane_body", None) is None
            ):
                return None
            return vec_width, inner_block_id, "unroll"
        return None
    # CuTe N-D tile strategy with lane loops: vec is set up per-block in
    # ``CuteNDTileStrategy.__init__`` when the autotuner picks
    # ``cute_vector_widths[block_id]`` > 1 and EPT is divisible by V.  Mode
    # is forced to ``"unroll"`` (per-element bitcast) for fp16/bf16 since
    # subscripting a bf16/fp16 vector in the CuTe DSL is unsafe; fp32
    # could in principle use ``"vec"`` but the per-element pipeline runs
    # most of the consume-sweep code after a cast, so unroll is the
    # robust choice.
    from ..tile_strategy import BlockSizeTileStrategy

    if isinstance(strategy, BlockSizeTileStrategy):
        vec_by_block = getattr(strategy, "_cute_lane_vec_width_by_block", None)
        if not isinstance(vec_by_block, dict):
            return None
        vec_width = vec_by_block.get(inner_block_id, 1)
        if vec_width <= 1:
            return None
        if not _cute_is_unroll_dtype(tensor.dtype):
            return None
        # The CuTe DSL's ``nvvm.load.ext`` ICEs at V=8 for fp16/bf16 (and
        # for the V=8 ``Uint8`` vector used by fp8), so widths > 4 cannot
        # use a single ``cute.arch.load``.  V=8 still
        # gets full LDG.128 throughput via the ``tile_unroll_split2``
        # mode: two back-to-back ``cute.arch.load(..., V=4)`` calls
        # (covering vec lanes 0-3 and 4-7) emit as two LDG.64s that the
        # SASS scheduler can overlap.  Wider Vs (16, 32, ...) are not
        # supported.
        if vec_width > 8:
            return None
        if vec_width == 8 and vec_width % 4 != 0:
            return None
        base_var_by_block = getattr(
            strategy, "_cute_lane_base_index_var_by_block", None
        )
        lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", None)
        vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", None)
        if (
            not isinstance(base_var_by_block, dict)
            or not isinstance(lane_body_by_block, dict)
            or not isinstance(vec_lane_var_by_block, dict)
            or inner_block_id not in base_var_by_block
            or inner_block_id not in lane_body_by_block
            or inner_block_id not in vec_lane_var_by_block
        ):
            return None
        # When the per-thread vec base could straddle the tensor edge
        # (e.g. ``numel`` not a multiple of V), the masked-tail iter
        # could load garbage in some lanes.  Gate the per-element mask
        # path correctly by requiring ``numel % V == 0`` so partial-vec
        # straddles are impossible.
        numel = env.block_sizes[inner_block_id].numel
        if not env.known_multiple(numel, vec_width):
            return None
        # Record the index_exprs position of the stride-1 lane axis so the
        # hoist substitutes the per-lane base there.  Row-major lhs loads
        # use the last position; a column-major rhs (K-major ``y``) uses
        # position 0.
        pos_by_block = getattr(strategy, "_cute_lane_axis_pos_by_block", None)
        if not isinstance(pos_by_block, dict):
            pos_by_block = {}
            # pyrefly: ignore [missing-attribute]
            strategy._cute_lane_axis_pos_by_block = pos_by_block
        pos_by_block[inner_block_id] = lane_axis_pos
        # fp8 loads a packed Uint64 (V=8) / Uint32 (V=4) in the regular
        # ``tile_unroll`` path — no ``VectorType`` so no V=8 ICE, hence no
        # split2 needed.  bf16/fp16 V=8 still needs the 2x V=4 split.
        if vec_width == 8 and not _cute_is_byte_packed(tensor.dtype):
            return vec_width, inner_block_id, "tile_unroll_split2"
        return vec_width, inner_block_id, "tile_unroll"
    return None


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

    from ...language import tile_index

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

    cute_state = state.device_function.cute_state
    if cute_state.suppress_root_lane_loops or (
        state.fx_node is not None
        and cute_state.is_collective_handled_load(state.fx_node.name)
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
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")
    if _is_cute_strided_slice_load_for_store(state, tensor, subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
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
    mask_expr = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor,
        include_tensor_index_masks=False,
    )
    vec_ctx = _cute_vector_load_ctx(state, tensor, subscript, index_exprs, extra_mask)
    if vec_ctx is not None:
        vec_width, vec_block_id, vec_mode = vec_ctx
        from ..reduction_strategy import LoopedReductionStrategy

        loops = state.codegen.active_device_loops.get(vec_block_id)
        strategy = loops[-1].strategy if loops else None
        if vec_mode == "vec":
            load_expr = _cute_vector_load_expr(
                tensor_name, index_exprs, tensor.dtype, vec_width=vec_width
            )
            # The mask is deferred to the post-fold scalar in
            # codegen_reduction.  The vec load itself is unconditional; the
            # mask is recorded on the active LoopedReductionStrategy and
            # applied around the folded sum.
            if isinstance(strategy, LoopedReductionStrategy):
                strategy._cute_emitted_vec_load = True
                if mask_expr is not None:
                    strategy._cute_pending_vec_masks.append(mask_expr)
            mask_expr = None
        elif vec_mode == "unroll":
            # Register (or reuse) a hoisted U16 vec load for this (tensor,
            # base_index) pair, then return ``hoist_var[vi].bitcast(dtype)``
            # so the existing scalar pipeline sees a scalar of the original
            # dtype.
            assert isinstance(strategy, LoopedReductionStrategy)
            load_expr = _cute_register_unroll_vec_hoist(
                state,
                strategy,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
        elif vec_mode == "tile_unroll":
            # Same hoist protocol as ``LoopedReductionStrategy``'s
            # ``unroll`` mode but for ``CuteNDTileStrategy`` lane loops.
            from ..tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
        else:
            assert vec_mode == "tile_unroll_split2"
            # V=8 fp16/bf16: emit two back-to-back ``cute.arch.load(...,
            # V=4)`` calls (lanes 0-3 and 4-7).  Works around the CuTe
            # DSL's ``nvvm.load.ext`` ICE on V=8 while still issuing the
            # full LDG.128 of bytes-per-thread-per-outer-iter.
            from ..tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist_split2(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
            )
    else:
        load_expr = _cute_scalar_load_expr(tensor_name, index_exprs, tensor.dtype)
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if state.fx_node is not None and _cute_load_feeds_sort_or_scan(state.fx_node):
        from .indexing import CuteSortableLoad

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
                else f"({load_expr} if {mask_expr} else {_cute_scalar_storage_dtype(tensor.dtype)}(0))"
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
    zero = _cute_scalar_storage_dtype(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")
