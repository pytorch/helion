"""Triton-backend codegen for ops defined in ``helion.language.quantized_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``quantized_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ...language.quantized_ops import load_float4_e2m1fn_x16_to_float16
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


_FP4X16_TO_F16_ASM = """
{
    .reg .b32 lo, hi;
    .reg .b8 c0,c1,c2,c3,c4,c5,c6,c7;
    .reg .b32 h0,h1,h2,h3,h4,h5,h6,h7;
    mov.b64 {lo, hi}, $16;
    mov.b32 {c0,c1,c2,c3}, lo;
    cvt.rn.f16x2.e2m1x2 h0, c0; cvt.rn.f16x2.e2m1x2 h1, c1;
    cvt.rn.f16x2.e2m1x2 h2, c2; cvt.rn.f16x2.e2m1x2 h3, c3;
    mov.b32 {c4,c5,c6,c7}, hi;
    cvt.rn.f16x2.e2m1x2 h4, c4; cvt.rn.f16x2.e2m1x2 h5, c5;
    cvt.rn.f16x2.e2m1x2 h6, c6; cvt.rn.f16x2.e2m1x2 h7, c7;
    mov.b32 {$0,$1}, h0; mov.b32 {$2,$3}, h1;
    mov.b32 {$4,$5}, h2; mov.b32 {$6,$7}, h3;
    mov.b32 {$8,$9}, h4; mov.b32 {$10,$11}, h5;
    mov.b32 {$12,$13}, h6; mov.b32 {$14,$15}, h7;
}
"""


@_decorators.codegen(load_float4_e2m1fn_x16_to_float16, "triton")
def _(state: CodegenState) -> list[ast.AST]:
    storage = state.proxy_arg(0)
    if not isinstance(storage, torch.Tensor):
        raise exc.InvalidAPIUsage(
            "hl.load_float4_e2m1fn_x16_to_float16 expects tensor storage"
        )
    group_offset = state.ast_arg(1)
    extra_mask = state.ast_args[2]
    base = state.device_function.tensor_arg(storage).name
    if extra_mask is None:
        load_call = expr_from_string(
            f"tl.load({base}.to(tl.pointer_type(tl.uint64)) + {{offset}})",
            offset=group_offset,
        )
    else:
        assert isinstance(extra_mask, ast.AST)
        mask_ast: ast.AST = extra_mask
        load_call = expr_from_string(
            f"tl.load({base}.to(tl.pointer_type(tl.uint64)) + {{offset}}, "
            "{mask}, other=0)",
            offset=group_offset,
            mask=mask_ast,
        )
    qword = state.codegen.lift(load_call, dce=True, prefix="fp4_qword")
    call = create(
        ast.Call,
        func=expr_from_string("tl.inline_asm_elementwise"),
        args=[
            create(ast.Constant, value=_FP4X16_TO_F16_ASM),
            create(ast.Constant, value=f"{','.join('=h' for _ in range(16))},l"),
            create(ast.List, elts=[qword], ctx=ast.Load()),
            expr_from_string("(" + ", ".join("tl.float16" for _ in range(16)) + ")"),
            create(ast.Constant, value=True),  # is_pure
            create(ast.Constant, value=1),  # pack
        ],
        keywords=[],
    )
    result = state.codegen.lift(call, dce=True, prefix="fp4_lanes")
    return [expr_from_string(f"{result.id}[{i}]") for i in range(16)]
