"""CuteDSL backend codegen registrations.

This module registers codegen functions for the CuteDSL backend.
Many operations share the Triton backend's codegen because they generate
backend-agnostic Python AST. Operations that use Triton-specific APIs
(tl.full, tl.where, tl.dot, etc.) have CuteDSL-specific implementations.

Import this module to register all CuteDSL codegen functions.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr

from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .variable_origin import BlockSizeOrigin
from ..language import _decorators

if TYPE_CHECKING:
    from .inductor_lowering import CodegenState

# ---------------------------------------------------------------------------
# Import the API functions we need to register codegen for
# ---------------------------------------------------------------------------
from ..language._tracing_ops import (  # noqa: E402
    _and,
    _constant_tensor,
    _for_loop,
    _get_symnode,
    _host_tensor,
    _if,
    _mask_to,
    _new_var,
    _not,
    _or,
    _phi,
    _while_loop,
)
from ..language.barrier import barrier  # noqa: E402
from ..language.constexpr import specialize  # noqa: E402
from ..language.creation_ops import full  # noqa: E402
from ..language.loops import grid  # noqa: E402
from ..language.loops import tile  # noqa: E402
from ..language.loops import _codegen_loop_helper  # noqa: E402
from ..language.matmul_ops import dot  # noqa: E402
from ..language.memory_ops import load  # noqa: E402
from ..language.memory_ops import store  # noqa: E402
from ..language.reduce_ops import _reduce  # noqa: E402
from ..language.tile_ops import tile_begin  # noqa: E402
from ..language.tile_ops import tile_count  # noqa: E402
from ..language.tile_ops import tile_end  # noqa: E402
from ..language.tile_ops import tile_id  # noqa: E402
from ..language.tile_ops import tile_index  # noqa: E402
from ..language.tile_ops import _disable_flatten_get_tile  # noqa: E402
from ..language.tunable_ops import register_block_size  # noqa: E402
from ..language.tunable_ops import register_tunable  # noqa: E402
from ..language.view_ops import join  # noqa: E402
from ..language.view_ops import split  # noqa: E402
from ..language.view_ops import subscript  # noqa: E402

BACKEND = "cutedsl"


# ===========================================================================
# Sharable codegen (no Triton-specific API calls)
# ===========================================================================

# --- loops ---
@_decorators.codegen(tile, BACKEND)
def _tile_codegen(state: CodegenState) -> ast.AST:
    return _codegen_loop_helper(state)


@_decorators.codegen(grid, BACKEND)
def _grid_codegen(state: CodegenState) -> ast.AST:
    return _codegen_loop_helper(state)


# --- tracing ops (backend-agnostic) ---
@_decorators.codegen(_get_symnode, BACKEND)
def _get_symnode_codegen(state: CodegenState) -> ast.AST:
    val = state.fx_node.meta["val"]
    if isinstance(val, int):
        return expr_from_string(str(val))
    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    sym_expr = val._sympy_()
    origin_info = HostFunction.current().expr_to_origin.get(sym_expr)
    if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
        block_size_var = state.device_function.block_size_var(
            origin_info.origin.block_id
        )
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.codegen(_host_tensor, BACKEND)
def _host_tensor_codegen(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")


@_decorators.codegen(_for_loop, BACKEND)
def _for_loop_codegen(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)


@_decorators.codegen(_while_loop, BACKEND)
def _while_loop_codegen(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(1)].codegen(state)


@_decorators.codegen(_if, BACKEND)
def _if_codegen(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(1)].codegen(state)


@_decorators.codegen(_phi, BACKEND)
def _phi_codegen(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs


@_decorators.codegen(_and, BACKEND)
def _and_codegen(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "{lhs} and {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.codegen(_or, BACKEND)
def _or_codegen(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "{lhs} or {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.codegen(_not, BACKEND)
def _not_codegen(state: CodegenState) -> ast.AST:
    return expr_from_string("not {lhs}", lhs=state.ast_arg(0))


@_decorators.codegen(_new_var, BACKEND)
def _new_var_codegen(state: CodegenState) -> ast.AST:
    value = state.ast_arg(0)
    assert isinstance(value, ast.AST)
    varname = state.codegen.tmpvar(
        prefix=value.id if isinstance(value, ast.Name) else "new_var"
    )
    state.add_statement(statement_from_string(f"{varname} = {{expr}}", expr=value))
    return create(ast.Name, id=varname, ctx=ast.Load())


# --- view ops ---
@_decorators.codegen(subscript, BACKEND)
def _subscript_codegen(state: CodegenState) -> ast.AST:
    output_keys = []
    for val in state.proxy_arg(1):
        if val is None:
            output_keys.append("None")
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_keys.append(":")
        else:
            from .. import exc
            raise exc.InvalidIndexingType(repr(val))
    return expr_from_string(
        f"{{base}}[{', '.join(output_keys)}]",
        base=state.ast_arg(0),
    )


# --- tile ops ---
@_decorators.codegen(tile_index, BACKEND)
def _tile_index_codegen(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.index_var(index))


@_decorators.codegen(tile_begin, BACKEND)
def _tile_begin_codegen(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.offset_var(index))


@_decorators.codegen(tile_id, BACKEND)
def _tile_id_codegen(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    offset = state.codegen.offset_var(index)
    block_size = state.device_function.block_size_var(index)
    if block_size is None:
        expr_str = offset
    else:
        expr_str = f"{offset} // {block_size}"
    return expr_from_string(expr_str)


# --- constexpr ---
@_decorators.codegen(specialize, BACKEND)
def _specialize_codegen(state: CodegenState) -> ast.AST:
    from ..language.constexpr import _convert_specializable
    value = state.proxy_arg(0)
    specialized = _convert_specializable(value)
    return expr_from_string(repr(specialized))


# --- tunable ops ---
@_decorators.codegen(register_block_size, BACKEND)
def _register_block_size_codegen(state: CodegenState) -> ast.AST:
    from ..language.tunable_ops import _block_id_from_state
    env = CompileEnvironment.current()
    block_size = env.config_spec.block_sizes.config_get(
        state.config.block_sizes, _block_id_from_state(state)
    )
    assert block_size is not None
    return expr_from_string(constant_repr(block_size))


@_decorators.codegen(register_tunable, BACKEND)
def _register_tunable_codegen(state: CodegenState) -> ast.AST:
    name = state.proxy_arg(0)
    assert isinstance(name, str)
    config_value = state.config[name]
    assert isinstance(config_value, (int, float, bool))
    return expr_from_string(constant_repr(config_value))


# --- barrier ---
@_decorators.codegen(barrier, BACKEND)
def _barrier_codegen(state: CodegenState) -> object:
    return expr_from_string("None")


# ===========================================================================
# CuteDSL-specific codegen (operations that use Triton-specific APIs)
# ===========================================================================

# --- memory ops (store/load) ---
# These delegate to the indexing strategy, which is the same infrastructure
# as Triton. The indexing strategy generates tl.store/tl.load calls which
# are Triton-specific, but for now we reuse them to get the backend wired up.
# A fully native CuteDSL memory system would be Phase 2+.

@_decorators.codegen(store, BACKEND)
def _store_codegen(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript_list = state.proxy_arg(1)
    assert isinstance(subscript_list, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        device_fn.device_store_index += 1
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_store(
            state, tensor, [*subscript_list], value, extra_mask
        )
    if isinstance(tensor, tuple):
        from .indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript_list], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.codegen(load, BACKEND)
def _load_codegen(state: CodegenState) -> ast.AST:
    from ..language.memory_ops import _EVICTION_POLICY_MAP

    tensor = state.proxy_arg(0)
    subscript_list = state.proxy_arg(1)
    assert isinstance(subscript_list, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function
    load_idx = device_fn.device_load_index
    device_fn.device_load_index += 1

    if eviction_policy is None and state.codegen.on_device:
        policies = state.config.load_eviction_policies
        if load_idx < len(policies):
            policy_value = policies[load_idx]
            eviction_policy = _EVICTION_POLICY_MAP.get(policy_value, policy_value)

    if eviction_policy is not None:
        assert isinstance(eviction_policy, str)
        eviction_policy = ast.Constant(value=eviction_policy)

    if isinstance(tensor, torch.Tensor):
        from ..language import tile_index as _tile_index_fn

        tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
        if (
            isinstance(tensor_node, torch.fx.Node)
            and tensor_node.op == "call_function"
            and tensor_node.target == _tile_index_fn
        ):
            env = CompileEnvironment.current()
            block_id = env.get_block_id(tensor.size(0))
            assert block_id is not None
            base_var = state.codegen.index_var(block_id)
            parts = []
            for idx in subscript_list:
                if idx is None:
                    parts.append("None")
                elif idx == slice(None):
                    parts.append(":")
                else:
                    raise AssertionError(
                        f"Unexpected index type in tile_index load: {idx}"
                    )
            return expr_from_string(f"{base_var}[{', '.join(parts)}]")

        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        return strategy.codegen_load(
            state, tensor, [*subscript_list], extra_mask, eviction_policy
        )
    if isinstance(tensor, tuple):
        from .indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state, tensor, dev_ptrs_ast, [*subscript_list], extra_mask, eviction_policy
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


# --- _constant_tensor (CuteDSL version) ---
# CuteDSL uses per-thread model: constants are just scalar values.
@_decorators.codegen(_constant_tensor, BACKEND)
def _constant_tensor_codegen(state: CodegenState) -> ast.AST:
    value = state.proxy_arg(0)
    dtype_str = state.proxy_arg(1)
    assert isinstance(value, (int, float, bool))
    assert isinstance(dtype_str, str)
    return expr_from_string(constant_repr(value))


# --- _mask_to (CuteDSL version) ---
# In per-thread model, mask is a scalar boolean. Use Python ternary.
@_decorators.codegen(_mask_to, BACKEND)
def _mask_to_codegen(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_exprs = []
    input_sizes = [*tensor.size()]
    for dim, size in enumerate(input_sizes):
        if (
            index := CompileEnvironment.current().resolve_block_id(size)
        ) is not None and (mask_var := state.codegen.mask_var(index)) is not None:
            mask_exprs.append(f"({mask_var})")
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = " and ".join(mask_exprs)
    other_repr = constant_repr(other)
    return expr_from_string(
        f"({{expr}} if {mask_expr} else {other_repr})",
        expr=state.ast_arg(0),
    )


# --- creation ops (full) ---
# In per-thread model, full is just a scalar value.
@_decorators.codegen(full, BACKEND)
def _full_codegen(state: CodegenState) -> ast.AST:
    proxy_value = state.proxy_arg(1)
    if isinstance(proxy_value, (int, float, bool)):
        value_str = state.device_function.literal_expr(proxy_value)
        return expr_from_string(value_str)
    value_ast = state.ast_arg(1)
    return value_ast


# --- tile_end (CuteDSL version — uses min()) ---
@_decorators.codegen(tile_end, BACKEND)
def _tile_end_codegen(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    offset_var = state.codegen.offset_var(index)
    block_size_var = state.device_function.block_size_var(index)
    if block_size_var is None:
        block_size_var = "1"
    naive_exp = f"{offset_var} + {block_size_var}"
    if state.codegen.mask_var(index) is not None:
        end_var = (
            state.codegen.active_device_loops[index][-1]
            .block_id_to_info[index]
            .end_var_name
        )
        return expr_from_string(f"min({naive_exp}, {end_var})")
    return expr_from_string(naive_exp)


# --- tile_count (CuteDSL version — uses cdiv expression) ---
@_decorators.codegen(tile_count, BACKEND)
def _tile_count_codegen(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    end_var = (
        state.codegen.active_device_loops[index][-1]
        .block_id_to_info[index]
        .end_var_name
    )
    block_size_var = state.device_function.block_size_var(index)
    if block_size_var is None:
        block_size_var = "1"
    return expr_from_string(
        f"(({end_var}) + ({block_size_var}) - 1) // ({block_size_var})"
    )


# --- view ops (split/join — CuteDSL versions) ---
@_decorators.codegen(split, BACKEND)
def _split_codegen(state: CodegenState) -> list[ast.AST]:
    raise NotImplementedError("split is not yet supported in CuteDSL backend")


@_decorators.codegen(join, BACKEND)
def _join_codegen(state: CodegenState) -> ast.AST:
    raise NotImplementedError("join is not yet supported in CuteDSL backend")


# --- reduce ops ---
@_decorators.codegen(_reduce, BACKEND)
def _reduce_codegen(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for reduce with combine function (CuteDSL version)."""
    from typing import cast as typing_cast
    from ..language.reduce_ops import _register_helper_function
    from ..language.reduce_ops import _create_reduce_expression
    from ..language.reduce_ops import _create_tuple_result_expressions

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    keep_dims = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    if is_tuple_input:
        input_tensor = state.ast_args[1]
        if isinstance(input_tensor, tuple):
            input_tensor = create(ast.Tuple, elts=list(input_tensor), ctx=ast.Load())
        else:
            input_tensor = state.ast_arg(1)
    else:
        input_tensor = state.ast_arg(1)
    helper_func_name = _register_helper_function(state, typing_cast(int, combine_graph_id))
    reduce_expr = _create_reduce_expression(
        input_tensor, dim, helper_func_name, bool(keep_dims)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, reduce_expr)
    return reduce_expr


# --- matmul ops (dot) ---
@_decorators.codegen(dot, BACKEND)
def _dot_codegen(state: CodegenState) -> object:
    """Generate code for dot product (CuteDSL version).

    Native CuteDSL MMA-based GEMM will be Phase 4.
    """
    raise NotImplementedError("dot is not yet supported in CuteDSL backend")
