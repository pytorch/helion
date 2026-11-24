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

from ..._compat import is_hip
from ...language import _decorators
from ...language.inline_asm_ops import inline_asm_elementwise
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..inline_asm_constraints import input_constraint_codes
from ..inline_asm_constraints import output_constraint_codes

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState

# NVPTX register classes. On ROCm, Triton's AMD backend logs an LLVM error for
# these but still builds the kernel, without the asm: results are silently
# wrong, so refuse them before Triton compiles.
_NVPTX_ONLY_CONSTRAINTS = frozenset("bcdfhlq")


def _check_rocm_constraints(constraints: str) -> None:
    backend = CompileEnvironment.current().backend
    outputs = output_constraint_codes(constraints)
    if not outputs:
        # Without an "=" output LLVM's AMDGPU backend aborts the process
        # (e.g. GCC's read-write "+v,v"); an input spelled "+v" is fine.
        raise backend.unsupported(
            f"inline asm constraints {constraints!r} declare no '=' output on ROCm"
        )
    codes = outputs + input_constraint_codes(constraints, allow_plus=True)
    if nvptx := sorted({code for code in codes if code in _NVPTX_ONLY_CONSTRAINTS}):
        raise backend.unsupported(
            f"inline asm constraint(s) {nvptx} in {constraints!r} on ROCm: these "
            "are NVPTX register classes; write AMDGCN asm with AMDGPU "
            "constraints such as 'v'"
        )


@_decorators.codegen(inline_asm_elementwise, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    # Get arguments
    asm_str = state.proxy_arg(0)
    constraints_str = state.proxy_arg(1)
    dtype = state.proxy_arg(3)
    is_pure = state.proxy_arg(4)
    pack = state.proxy_arg(5)
    # ROCm specifically: Triton's AMD backend is what drops this asm silently.
    # Other targets that do not lower to PTX (TileIR) reject inline asm.
    if is_hip():
        assert isinstance(constraints_str, str)  # checked by the language op
        _check_rocm_constraints(constraints_str)

    # Convert the list of tensor args to AST
    # We need to create a proper list AST with the tensor elements
    raw_args = state.ast_args[2]
    if isinstance(raw_args, (list, tuple)):
        # FX containers are immutable; AST transforms mutate their list fields.
        args_ast = create(ast.List, elts=list(raw_args), ctx=ast.Load())
    else:
        # Preserve an argument that is already represented by an AST expression.
        args_ast = raw_args

    # Convert dtype to Triton type string(s)
    if isinstance(dtype, (tuple, list)):
        dtype_strs = [triton_type(dt) for dt in dtype if isinstance(dt, torch.dtype)]
        dtype_arg = create(
            ast.Tuple,
            elts=[expr_from_string(value) for value in dtype_strs],
            ctx=ast.Load(),
        )
        has_multiple_outputs = True
    else:
        dtype_arg = expr_from_string(
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
            dtype_arg,
            create(ast.Constant, value=is_pure),
            create(ast.Constant, value=pack),
        ],
        keywords=[],
    )

    # Handle multiple outputs by creating getitem expressions of one call,
    # which asm not declared pure keeps even when nothing reads them
    if has_multiple_outputs:
        assert isinstance(dtype, (tuple, list))  # Type guard for len()
        return state.tuple_results(
            inline_asm_call,
            len(dtype),
            prefix="inline_asm_result",
            dce=is_pure is True,
        )

    return inline_asm_call
