"""Triton-backend codegen for ops defined in ``helion.language.quantized_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``quantized_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ..ast_extension import create
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


# Triton uses inline asm for the same fp4x2 unpack instruction as CuTe.
_FP4X2_TO_F32_ASM = """
{
    .reg .b8 b0;
    .reg .b16 lo, hi;
    .reg .b32 h2;
    mov.b16 {b0, _}, $2;
    cvt.rn.f16x2.e2m1x2 h2, b0;
    mov.b32 {lo, hi}, h2;
    cvt.f32.f16 $0, lo;
    cvt.f32.f16 $1, hi;
}
"""


@_decorators.codegen(float4_e2m1fn_x2_to_float32, "triton")
def _(state: CodegenState) -> list[ast.AST]:
    value = state.codegen.lift(
        expr_from_string("{value}.to(tl.int16)", value=state.ast_arg(0)),
        dce=True,
        prefix="fp4_packed",
    )
    call = create(
        ast.Call,
        func=expr_from_string("tl.inline_asm_elementwise"),
        args=[
            create(ast.Constant, value=_FP4X2_TO_F32_ASM),
            create(ast.Constant, value="=f,=f,h"),
            create(ast.List, elts=[value], ctx=ast.Load()),
            expr_from_string("(tl.float32, tl.float32)"),
            create(ast.Constant, value=True),  # is_pure
            create(ast.Constant, value=1),  # pack
        ],
        keywords=[],
    )
    result = state.codegen.lift(call, dce=True, prefix="fp4_pair")
    return [
        expr_from_string(f"{result.id}[0]"),
        expr_from_string(f"{result.id}[1]"),
    ]
