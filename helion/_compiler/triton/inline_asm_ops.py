"""Triton-backend codegen for the ops defined in ``helion.language.inline_asm_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``inline_asm_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from ...language import _decorators
from ...language.inline_asm_ops import inline_asm_elementwise
from ..ast_extension import create
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(inline_asm_elementwise, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    # Get arguments
    asm_str = state.proxy_arg(0)
    constraints_str = state.proxy_arg(1)
    dtype = state.proxy_arg(3)
    is_pure = state.proxy_arg(4)
    pack = state.proxy_arg(5)

    # Convert the list of tensor args to AST
    # We need to create a proper list AST with the tensor elements
    raw_args = state.ast_args[2]
    if isinstance(raw_args, list):
        # Create AST List node with the tensor elements
        args_ast = create(ast.List, elts=raw_args, ctx=ast.Load())
    else:
        # If it's not a list, wrap it in a list (shouldn't normally happen)
        args_ast = raw_args

    # Convert dtype to Triton type string(s)
    if isinstance(dtype, (tuple, list)):
        dtype_strs = [triton_type(dt) for dt in dtype if isinstance(dt, torch.dtype)]
        dtype_arg = f"({', '.join(dtype_strs)})"  # Use tuple syntax for multiple dtypes
        has_multiple_outputs = True
    else:
        dtype_arg = (
            triton_type(dtype) if isinstance(dtype, torch.dtype) else "tl.float32"
        )
        has_multiple_outputs = False

    # Create the call to tl.inline_asm_elementwise
    inline_asm_call = create(
        ast.Call,
        func=expr_from_string("tl.inline_asm_elementwise"),
        args=[
            create(ast.Constant, value=asm_str),
            create(ast.Constant, value=constraints_str),
            args_ast,
            expr_from_string(dtype_arg),
            create(ast.Constant, value=is_pure),
            create(ast.Constant, value=pack),
        ],
        keywords=[],
    )

    # Handle multiple outputs by creating getitem expressions
    if has_multiple_outputs:
        assert isinstance(dtype, (tuple, list))  # Type guard for len()
        num_outputs = len(dtype)
        return [
            expr_from_string(
                f"{{inline_asm_result}}[{i}]", inline_asm_result=inline_asm_call
            )
            for i in range(num_outputs)
        ]

    return inline_asm_call
