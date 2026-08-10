"""Triton-backend codegen for the atomic ops defined in ``helion.language.atomic_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``atomic_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.atomic_ops import _to_ast_values
from ...language.atomic_ops import atomic_add
from ...language.atomic_ops import atomic_and
from ...language.atomic_ops import atomic_cas
from ...language.atomic_ops import atomic_max
from ...language.atomic_ops import atomic_min
from ...language.atomic_ops import atomic_or
from ...language.atomic_ops import atomic_xchg
from ...language.atomic_ops import atomic_xor
from ..ast_extension import expr_from_string
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


def _codegen_common(
    op: str, state: CodegenState, value_exprs: list[ast.AST]
) -> ast.AST:
    """Route any single-value atomic op through the atomic_indexing strategy."""
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(op)

    device_fn = state.device_function
    fx_node = state.fx_node
    epilogue_subtile_group_id = (
        None if fx_node is None else fx_node.meta.get("epilogue_subtile_group_id")
    )
    if epilogue_subtile_group_id is None:
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
    elif fx_node is not None and fx_node.meta.get(
        "epilogue_subtile_primary_output", False
    ):
        indexing_idx = device_fn.atomic_op_index
        device_fn.atomic_op_index += 1
        device_fn.epilogue_subtile_atomic_indices[epilogue_subtile_group_id] = (
            indexing_idx
        )
    else:
        indexing_idx = device_fn.epilogue_subtile_atomic_indices[
            epilogue_subtile_group_id
        ]
    strategy = device_fn.get_atomic_indexing_strategy(indexing_idx)
    return strategy.codegen_atomic(op, state, target, index, value_exprs[0], sem)


@_decorators.codegen(atomic_add, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_add", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xchg, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xchg", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_and, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_and", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_or, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_or", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_xor, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xor", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_max, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_max", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_min, "triton")
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_min", state, _to_ast_values([value_expr]))


@_decorators.codegen(atomic_cas, "triton")
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    # CAS always uses pointer (not a TMA reduction op, two values),
    # but increment the counter to keep per-op atomic_indexing aligned.
    device_fn = state.device_function
    device_fn.atomic_op_index += 1

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        f"tl.atomic_cas({name} + {{offset}}, {{exp}}, {{val}}, sem={{sem}})",
        offset=indices.index_expr,
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )
