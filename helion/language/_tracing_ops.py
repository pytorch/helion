from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeVar

import sympy
import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.utils import triton_type
from torch.fx import has_side_effect
from torch.fx.experimental.sym_node import SymNode

from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import _can_epilogue_subtile_with_output_shape
from .._compiler.indexing_strategy import _get_output_shape
from .._compiler.indexing_strategy import _get_subtile_split
from .._compiler.utils import _use_epilogue_subtile
from ..exc import NotInsideKernel
from . import _decorators
from .tile_proxy import Tile
from helion._compiler.indexing_strategy import BlockedSubscriptIndexing
from helion._compiler.indexing_strategy import PointerIndexingStrategy
from helion._compiler.indexing_strategy import SubscriptIndexing
from helion._compiler.indexing_strategy import TensorDescriptorIndexingStrategy

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T", bound=object)

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""

_symbolic_types = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)


@_decorators.api()
def _get_symnode(debug_name: str) -> int:
    """FX requires a torch.SymInt to come from an op. This is a fake op is added lazily to work around this."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_get_symnode)
def _(state: CodegenState) -> ast.AST:
    val = state.fx_node.meta["val"]  # pyright: ignore[reportOptionalMemberAccess]

    # Handle the case where val is a regular integer (e.g., from reduction_loops config)
    if isinstance(val, int):
        return expr_from_string(str(val))

    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    if (block_idx := CompileEnvironment.current().get_block_id(val)) is not None:  # pyright: ignore[reportArgumentType]
        block_size_var = state.device_function.block_size_var(block_idx)
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    sym_expr = val._sympy_()
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.api()
def _host_tensor(debug_name: str) -> torch.Tensor:
    """Source of a tensor that was allocated on the host and must be passed to the kernel as an arg."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")  # should be unused


@has_side_effect
@_decorators.api()
def _for_loop(
    graph_id: int, begin: list[int], end: list[int], args: list[object]
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop)
def _(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)  # pyright: ignore[reportArgumentType,reportCallIssue]


@has_side_effect
@_decorators.api()
def _if(test: object, graph_id: int, args: list[object]) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_if)
def _(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(1)].codegen(state)  # pyright: ignore[reportArgumentType,reportCallIssue]


# Note we can't DCE phi nodes because there may be a loop carry dependency not captured in the outer graph
@has_side_effect
@_decorators.api(allow_host_tensor=True)
def _phi(lhs: object, rhs: object) -> object:
    """Combine values from different branches of a control flow."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_phi)
def _(lhs: object, rhs: object) -> object:
    if isinstance(lhs, Tile):
        assert isinstance(rhs, Tile)
        assert lhs.block_id == rhs.block_id
        return lhs
    assert isinstance(lhs, torch.Tensor), lhs
    assert isinstance(rhs, torch.Tensor), rhs
    assert lhs.size() == rhs.size()
    assert lhs.dtype == rhs.dtype
    assert lhs.device == rhs.device
    return torch.empty_like(lhs)


@_decorators.codegen(_phi)
def _(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs


@_decorators.get_masked_value(_phi)
def _(node: torch.fx.Node) -> float | bool | None:
    lhs, rhs = node.args
    assert isinstance(lhs, torch.fx.Node)
    assert isinstance(rhs, torch.fx.Node)

    from .._compiler.node_masking import cached_masked_value

    lval = cached_masked_value(lhs)
    if lval is not None:
        rval = cached_masked_value(rhs)
        if lval == rval:
            return lval
    return None


@_decorators.api()
def _inductor_lowering_extra(args: list[object]) -> torch.Tensor:
    """
    When we have an inductor lowering that results in multiple inductor
    buffers, we insert this fake op in the graph to represent intermediate
    values.
    """
    raise AssertionError("this should never be called")


@_decorators.api()
def _and(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.codegen(_and)
def _(state: CodegenState) -> None:
    return expr_from_string(
        "{lhs} and {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )  # pyright: ignore[reportReturnType]


@_decorators.register_fake(_and)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if not left:
            return left
        return right
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.And(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    # TODO(jansel): should match the type of the input
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.api()
def _or(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_or)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if left:
            return left
        return right
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.Or(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_or)
def _(state: CodegenState) -> None:
    return expr_from_string(
        "{lhs} or {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )  # pyright: ignore[reportReturnType]


@_decorators.api()
def _not(left: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_not)
def _(left: object) -> object:
    if not isinstance(left, _symbolic_types):
        return not left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool):
        return torch.SymBool(
            SymNode(sympy.Not(left._sympy_()), env.shape_env, bool, hint=None)
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_not)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "not {lhs}",
        lhs=state.ast_arg(0),
    )


@_decorators.api()
def _mask_to(tensor: torch.Tensor, other: float | bool, /) -> torch.Tensor:
    """
    Set the masked out values of a given tile to a specific value.
    This operation is automatically generated by the compiler when doing a
    dot or reduction operation, and should not need to be called directly
    by users.

    Args:
        tensor: The tensor to apply the mask to.
        other: The value to set the masked out elements to.

    Returns:
        torch.Tensor: A tensor with the masked out elements set to `other`.
    """
    raise NotInsideKernel


@_decorators.register_fake(_mask_to)
def _(tensor: torch.Tensor, other: float) -> torch.Tensor:
    return torch.empty_like(tensor)


@_decorators.codegen(_mask_to)
def _(state: CodegenState) -> ast.AST:
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
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            mask_exprs.append(f"({mask_var}{expand})")
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = "&".join(mask_exprs)
    if len(mask_exprs) < len(input_sizes):
        mask_expr = f"tl.broadcast_to({mask_expr}, {state.tile_strategy.shape_str(input_sizes)})"
    # Ensure the masked value literal matches the tensor dtype to avoid unintended upcasts
    input_dtype = tensor.dtype
    other_typed = expr_from_string(
        f"tl.full([], {constant_repr(other)}, {triton_type(input_dtype)})"
    )
    return expr_from_string(
        f"tl.where({mask_expr}, {{expr}}, {{other}})",
        expr=state.ast_arg(0),
        other=other_typed,
    )


@_decorators.get_masked_value(_mask_to)
def _(node: torch.fx.Node) -> float | bool:
    value = node.args[1]
    assert isinstance(value, (int, float, bool))
    return value


@_decorators.api(allow_host_tensor=True)
def _new_var(value: _T, /) -> _T:
    """
    Create a shallow copy of a value that is assigned a fresh variable in codegen.

    This is used to ensure phi() node handling works properly when a value is renamed
    without mutation in a loop.  We need to copy the inputs to a loop so that phi nodes
    are handled properly.  Phi nodes will merge variable names from outside the loop,
    but the old value of those variables could have usages.
    """
    raise NotInsideKernel


@_decorators.register_fake(_new_var)
def _(value: _T) -> _T:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value)
    if isinstance(value, torch.SymInt):
        return CompileEnvironment.current().create_unbacked_symint()  # pyright: ignore[reportReturnType]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    raise NotImplementedError(f"Unsupported type for _new_var: {type(value)}")


@_decorators.codegen(_new_var)
def _(state: CodegenState) -> ast.AST:
    value = state.ast_arg(0)
    assert isinstance(value, ast.AST)
    varname = state.codegen.tmpvar(
        prefix=value.id if isinstance(value, ast.Name) else "new_var"
    )
    state.add_statement(statement_from_string(f"{varname} = {{expr}}", expr=value))
    return create(ast.Name, id=varname, ctx=ast.Load())


@_decorators.get_masked_value(_new_var)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    (arg,) = node.args
    assert isinstance(arg, torch.fx.Node)
    return cached_masked_value(arg)


@has_side_effect
@_decorators.api(allow_host_tensor=True)
def _subtile_store(
    tensor: torch.Tensor,
    subscript: list[object],
    value: torch.Tensor,
) -> None:
    """
    Internal op for epilogue subtiled stores.

    This operation replaces hl.store when epilogue subtiling is enabled.
    During codegen, it reads the subtile split factor from config and either:
    - Emits a normal store if split factor <= 0 or subtiling not applicable
    - Splits the value and emits two stores with modified indices

    This operation is automatically generated by the epilogue_subtiling_pass
    and should not need to be called directly by users.
    """
    raise NotInsideKernel


@_decorators.codegen(_subtile_store)
def _(state: CodegenState) -> ast.AST | None:
    fake_tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    value_ast = state.ast_arg(2)

    assert isinstance(fake_tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    # Get the indexing strategy for this store
    device_fn = state.device_function
    strategy_index = device_fn.device_memory_op_index
    strategy = device_fn.get_indexing_strategy(strategy_index)
    device_fn.device_memory_op_index += 1
    device_fn.device_store_index += 1

    # Check if we should use subtiling
    subtile_split = _get_subtile_split(state)

    if not _use_epilogue_subtile() or subtile_split is None:
        # Fall back to normal store - pointwise operations already ran during normal codegen
        return strategy.codegen_store(state, fake_tensor, [*subscript], value_ast, None)

    # From here on, subtiling is active, so pointwise ops were skipped and need to be applied to subtiles

    # Determine if subtiling is applicable based on indexing strategy
    if isinstance(strategy, TensorDescriptorIndexingStrategy):
        if not TensorDescriptorIndexingStrategy.is_supported(
            state, fake_tensor, [*subscript], None
        ):
            # Fall back to pointer store, apply pointwise since we're subtiling
            (value_with_pointwise,) = _apply_pointwise_to_subtiles(state, [value_ast])
            return PointerIndexingStrategy().codegen_store(
                state, fake_tensor, [*subscript], value_with_pointwise, None
            )
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, [*subscript])
        output_shape = _get_output_shape(indexing, state)
    elif isinstance(strategy, PointerIndexingStrategy):
        indexing = SubscriptIndexing.create(state, fake_tensor, [*subscript], None)
        output_shape = _get_output_shape(indexing, state, fake_tensor, [*subscript])
    else:
        # BlockPtr or other strategy - fall back, apply pointwise since we're subtiling
        (value_with_pointwise,) = _apply_pointwise_to_subtiles(state, [value_ast])
        return strategy.codegen_store(state, fake_tensor, [*subscript], value_with_pointwise, None)

    if not _can_epilogue_subtile_with_output_shape(output_shape):
        # Subtiling not applicable - fall back, apply pointwise since we're subtiling
        (value_with_pointwise,) = _apply_pointwise_to_subtiles(state, [value_ast])
        return strategy.codegen_store(state, fake_tensor, [*subscript], value_with_pointwise, None)

    # Apply subtiling
    return _codegen_subtile_store(
        state,
        fake_tensor,
        subscript,
        value_ast,
        indexing,
        output_shape,
        subtile_split,
        strategy,
    )


def _codegen_subtile_store(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: object,
    value_ast: ast.AST,
    indexing: object,
    output_shape: list[int | torch.SymInt],
    subtile_split: int,
    strategy: object,
) -> ast.AST:
    """Generate the subtiled store operations."""

    codegen = state.codegen
    device_fn = state.device_function
    env = CompileEnvironment.current()

    block_m, block_n = output_shape
    block_n_str = device_fn.literal_expr(block_n)
    block_n_half_str = f"({block_n_str} // {subtile_split})"
    block_m_str = device_fn.literal_expr(block_m)

    # Reshape and split the accumulator
    acc_var = codegen.lift(value_ast, prefix="acc")
    reshape_expr = expr_from_string(
        "tl.reshape({acc}, [{dim_m}, 2, {dim_half}]).permute(0, 2, 1)",
        acc=acc_var,
        dim_m=expr_from_string(block_m_str),
        dim_half=expr_from_string(block_n_half_str),
    )
    reshape_var = codegen.lift(reshape_expr, prefix="acc")

    acc0_name = codegen.tmpvar(prefix="acc")
    acc1_name = codegen.tmpvar(prefix="acc")
    codegen.add_statement(
        statement_from_string(
            f"{acc0_name}, {acc1_name} = tl.split({{acc}})",
            acc=reshape_var,
        )
    )

    # Apply pointwise operations to each subtile if present
    acc0 = expr_from_string(acc0_name)
    acc1 = expr_from_string(acc1_name)
    acc0, acc1 = _apply_pointwise_to_subtiles(state, [acc0, acc1])

    # Get block indices
    block_idx = env.get_block_id(block_n)
    block_idx_m = env.get_block_id(block_m)

    if block_idx is None or block_idx_m is None:
        # Fall back to normal store if we can't get block indices
        from .._compiler.indexing_strategy import PointerIndexingStrategy

        return PointerIndexingStrategy().codegen_store(
            state, fake_tensor, [*subscript], value_ast, None  # pyright: ignore[reportGeneralTypeIssues]
        )

    # Generate stores based on indexing type
    if isinstance(indexing, BlockedSubscriptIndexing):
        return _codegen_tensor_descriptor_subtile_stores(
            state,
            fake_tensor,
            indexing,
            acc0,
            acc1,
            block_n_half_str,
            subtile_split,
        )
    assert isinstance(indexing, SubscriptIndexing)
    return _codegen_pointer_subtile_stores(
        state,
        fake_tensor,
        subscript,
        indexing,
        acc0,
        acc1,
        block_idx,
        block_idx_m,
        block_n_half_str,
    )


def _apply_pointwise_to_subtiles(
    state: CodegenState, tile_values: list[ast.AST]
) -> list[ast.AST]:
    """Apply pointwise operations to each subtile if present."""
    from torch._inductor import ir

    from .._compiler.inductor_lowering import PointwiseLowering, install_inductor_kernel_handlers

    if not (
        isinstance(state.fx_node, torch.fx.Node)
        and "pointwise_epilogue_nodes" in state.fx_node.meta
    ):
        return tile_values

    pointwise_nodes = list(reversed(state.fx_node.meta["pointwise_epilogue_nodes"]))
    for pw_node in pointwise_nodes:
        lowering = pw_node.meta["lowering"]
        assert isinstance(lowering, PointwiseLowering)

        buffer = lowering.buffer
        assert isinstance(buffer.data, ir.Pointwise)

        for i, tile in enumerate(tile_values):
            codegen = state.codegen
            subtile_var = codegen.lift(tile, prefix="subtile")

            with install_inductor_kernel_handlers(
                codegen, dict.fromkeys(lowering.input_names, subtile_var)
            ):
                # Generate the pointwise operation
                indices = [
                    sympy.Symbol(f"i{n}") for n in range(len(buffer.data.ranges))
                ]
                from .._compiler.inductor_lowering import _unpack_opsvalue

                result_name = _unpack_opsvalue(buffer.data.inner_fn(indices))
                tile_values[i] = expr_from_string(result_name)

    return tile_values


def _codegen_tensor_descriptor_subtile_stores(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    indexing: object,  # BlockedSubscriptIndexing
    acc0: ast.AST,
    acc1: ast.AST,
    block_n_half_str: str,
    subtile_split: int,
) -> ast.AST:
    """Generate tensor descriptor subtile stores."""
    from .._compiler.indexing_strategy import BlockedSubscriptIndexing

    assert isinstance(indexing, BlockedSubscriptIndexing)
    codegen = state.codegen

    # Modify block shape for subtiles
    indexing.block_shape[1] //= subtile_split

    desc_name = indexing.tensor_descriptor(state)
    offset0 = expr_from_string(indexing.offsets[0])
    offset1 = expr_from_string(indexing.offsets[1])

    # First subtile store
    codegen.add_statement(
        statement_from_string(
            f"{desc_name}.store([{{off0}}, {{off1}}], {{value}})",
            off0=offset0,
            off1=offset1,
            value=acc0,
        )
    )

    # Second subtile store with shifted offset
    offset1_shifted = expr_from_string(
        "({offset} + {half})",
        offset=expr_from_string(indexing.offsets[1]),
        half=expr_from_string(block_n_half_str),
    )

    return expr_from_string(
        f"{desc_name}.store([{{off0}}, {{off1}}], {{value}})",
        off0=offset0,
        off1=offset1_shifted,
        value=acc1,
    )


def _codegen_pointer_subtile_stores(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    subscript: object,
    indexing: object,  # SubscriptIndexing
    acc0: ast.AST,
    acc1: ast.AST,
    block_idx: int,
    block_idx_m: int,
    block_n_half_str: str,
) -> ast.AST:
    """Generate pointer-based subtile stores."""
    from .._compiler.indexing_strategy import SubscriptIndexing

    assert isinstance(indexing, SubscriptIndexing)
    codegen = state.codegen
    device_fn = state.device_function

    name = device_fn.tensor_arg(fake_tensor).name
    offset_n_var = codegen.offset_var(block_idx)

    # Create sliced indices for each subtile
    index_n_0_name = codegen.tmpvar(prefix="indices_n")
    codegen.add_statement(
        statement_from_string(
            f"{index_n_0_name} = ({offset_n_var} + tl.arange(0, {block_n_half_str})).to(tl.int32)"
        )
    )

    index_n_1_name = codegen.tmpvar(prefix="indices_n")
    codegen.add_statement(
        statement_from_string(
            f"{index_n_1_name} = ({offset_n_var} + {block_n_half_str} + tl.arange(0, {block_n_half_str})).to(tl.int32)"
        )
    )

    # Reconstruct offset expressions for each subtile
    stride_n = device_fn.tensor_stride(fake_tensor, -1).name
    stride_m = device_fn.tensor_stride(fake_tensor, -2).name
    index_m_var = codegen.index_var(block_idx_m)

    offset_0 = expr_from_string(
        f"{index_m_var}[:, None] * {stride_m} + {index_n_0_name}[None, :] * {stride_n}"
    )
    offset_1 = expr_from_string(
        f"{index_m_var}[:, None] * {stride_m} + {index_n_1_name}[None, :] * {stride_n}"
    )

    # Generate masks for each subtile if masking is needed
    mask_0 = indexing.mask_expr
    mask_1 = indexing.mask_expr

    if indexing.has_mask():
        mask_n_var = codegen.mask_var(block_idx)
        if mask_n_var is not None:
            mask_n_0_name = codegen.tmpvar(prefix="mask_n")
            mask_n_1_name = codegen.tmpvar(prefix="mask_n")

            codegen.add_statement(
                statement_from_string(
                    f"{mask_n_0_name} = {index_n_0_name} < {stride_m}"
                )
            )
            codegen.add_statement(
                statement_from_string(
                    f"{mask_n_1_name} = {index_n_1_name} < {stride_m}"
                )
            )

            mask_m_var = codegen.mask_var(block_idx_m)
            if mask_m_var is not None:
                mask_0 = expr_from_string(
                    f"{mask_m_var}[:, None] & {mask_n_0_name}[None, :]"
                )
                mask_1 = expr_from_string(
                    f"{mask_m_var}[:, None] & {mask_n_1_name}[None, :]"
                )
            else:
                mask_0 = expr_from_string(f"{mask_n_0_name}[None, :]")
                mask_1 = expr_from_string(f"{mask_n_1_name}[None, :]")

    # First subtile store
    codegen.add_statement(
        statement_from_string(
            f"tl.store({name} + {{offset}}, {{value}}, {{mask}})",
            value=acc0,
            offset=offset_0,
            mask=mask_0,
        )
    )

    # Second subtile store - return as the result
    return expr_from_string(
        f"tl.store({name} + {{offset}}, {{value}}, {{mask}})",
        value=acc1,
        offset=offset_1,
        mask=mask_1,
    )

