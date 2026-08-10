"""CuTe-backend codegen for the ops defined in ``helion.language.inline_asm_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``inline_asm_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.inline_asm_ops import inline_asm_elementwise
from ..ast_extension import create
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(inline_asm_elementwise, "cute")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    asm_str = state.proxy_arg(0)
    constraints_str = state.proxy_arg(1)
    dtype = state.proxy_arg(3)
    is_pure = state.proxy_arg(4)
    pack = state.proxy_arg(5)

    if pack != 1:
        raise exc.BackendUnsupported(
            "cute",
            "hl.inline_asm_elementwise with pack != 1",
        )

    raw_args = state.ast_args[2]
    if isinstance(raw_args, list):
        args_ast = create(ast.Tuple, elts=raw_args, ctx=ast.Load())
    elif isinstance(raw_args, tuple):
        args_ast = create(ast.Tuple, elts=list(raw_args), ctx=ast.Load())
    else:
        args_ast = create(ast.Tuple, elts=[raw_args], ctx=ast.Load())

    from ..compile_environment import CompileEnvironment

    backend = CompileEnvironment.current().backend
    if isinstance(dtype, (tuple, list)):
        dtype_arg = create(
            ast.Tuple,
            elts=[
                expr_from_string(backend.dtype_str(dt))
                for dt in dtype
                if isinstance(dt, torch.dtype)
            ],
            ctx=ast.Load(),
        )
        has_multiple_outputs = True
    else:
        dtype_arg = expr_from_string(
            backend.dtype_str(dtype)
            if isinstance(dtype, torch.dtype)
            else "cutlass.Float32"
        )
        has_multiple_outputs = False

    inline_asm_call = create(
        ast.Call,
        func=expr_from_string("_cute_inline_asm_elementwise"),
        args=[args_ast],
        keywords=[
            create(ast.keyword, arg="asm", value=create(ast.Constant, value=asm_str)),
            create(
                ast.keyword,
                arg="constraints",
                value=create(ast.Constant, value=constraints_str),
            ),
            create(ast.keyword, arg="dtype", value=dtype_arg),
            create(
                ast.keyword,
                arg="is_pure",
                value=create(ast.Constant, value=is_pure),
            ),
        ],
    )

    if has_multiple_outputs:
        assert isinstance(dtype, (tuple, list))
        inline_asm_result = state.codegen.lift(
            inline_asm_call, dce=True, prefix="inline_asm_result"
        )
        return [
            expr_from_string(f"{inline_asm_result.id}[{i}]") for i in range(len(dtype))
        ]

    return inline_asm_call
