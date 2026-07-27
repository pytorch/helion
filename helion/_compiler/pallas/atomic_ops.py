"""Pallas-backend codegen for the atomic ops defined in ``helion.language.atomic_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
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
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


def _pallas_atomic_load_prev(
    state: CodegenState,
) -> tuple[str, str, str]:
    """Load previous value for a Pallas atomic op.

    On TPU, each kernel instance has exclusive access to its tile, so
    atomics are implemented as regular load-compute-store sequences.

    Returns (tensor_name, index_str, prev_var_name).
    """
    from . import codegen as pallas_codegen

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = pallas_codegen.index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )
    return name, index_str, prev_var  # pyrefly: ignore[bad-return]


@_decorators.codegen(atomic_add, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    target = state.proxy_arg(0)
    assert isinstance(target, torch.Tensor)
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(target.dtype)
    # Cast the sum to the target dtype so the store doesn't fail when
    # the value dtype differs (e.g. float32 accumulator into bfloat16 ref).
    cast = backend.cast_expr(f"{prev_var} + {{value}}", target_dtype)
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {cast}", value=value_ast)
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_xchg, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(f"{name}[{index_str}] = {{value}}", value=value_ast)
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_and, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} & {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_or, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} | {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_xor, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = {prev_var} ^ {{value}}", value=value_ast
        )
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_max, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.maximum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_min, "pallas")
def _(state: CodegenState) -> ast.AST:
    name, index_str, prev_var = _pallas_atomic_load_prev(state)
    value_ast = _to_ast_values([state.ast_args[2]])[0]
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.minimum({prev_var}, {{value}})",
            value=value_ast,
        )
    )
    return expr_from_string(prev_var)


@_decorators.codegen(atomic_cas, "pallas")
def _(state: CodegenState) -> ast.AST:
    from . import codegen as pallas_codegen

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list, tuple))

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor("pallas atomic_cas")

    name = state.device_function.tensor_arg(target).name
    index_str, _ = pallas_codegen.index_str(state, index, target)

    prev_var = state.device_function.new_var("_prev", dce=True)
    state.codegen.add_statement(
        statement_from_string(f"{prev_var} = {name}[{index_str}]")
    )

    exp_ast, val_ast = _to_ast_values([state.ast_args[2], state.ast_args[3]])
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{index_str}] = jnp.where({prev_var} == {{exp}}, {{val}}, {prev_var})",
            exp=exp_ast,
            val=val_ast,
        )
    )
    return expr_from_string(prev_var)
