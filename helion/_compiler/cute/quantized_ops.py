"""CuTe-backend codegen for ops defined in ``helion.language.quantized_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``quantized_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ...language.quantized_ops import load_bfloat16_x16_to_float16
from ...language.quantized_ops import load_float4_e2m1fn_x16_to_float16
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(float4_e2m1fn_x2_to_float32, "cute")
def _(state: CodegenState) -> list[ast.AST]:
    call = expr_from_string(
        "_cute_float4_e2m1fn_x2_to_float32({value})",
        value=state.ast_arg(0),
    )
    result = state.codegen.lift(call, dce=True, prefix="fp4_pair")
    return [
        expr_from_string(f"{result.id}[0]"),
        expr_from_string(f"{result.id}[1]"),
    ]


@_decorators.codegen(load_float4_e2m1fn_x16_to_float16, "cute")
def _(state: CodegenState) -> list[ast.AST]:
    storage = state.proxy_arg(0)
    if not isinstance(storage, torch.Tensor):
        raise TypeError("load_float4_e2m1fn_x16_to_float16 storage must be a tensor")

    base = state.device_function.tensor_arg(storage).name
    group_offset = state.ast_arg(1)
    extra_mask = state.ast_args[2]
    load = expr_from_string(
        f"cute.arch.load({base}.iterator + ({{offset}} * 8), cutlass.Uint64)",
        offset=group_offset,
    )
    if extra_mask is not None:
        assert isinstance(extra_mask, ast.AST)
        load = expr_from_string(
            "({load} if {mask} else cutlass.Uint64(0))",
            load=load,
            mask=extra_mask,
        )
    qword = state.codegen.lift(load, dce=True, prefix="fp4_qword")
    call = expr_from_string(
        "_cute_float4_e2m1fn_x16_to_float16({value})",
        value=qword,
    )
    result = state.codegen.lift(call, dce=True, prefix="fp4_lanes")
    return [expr_from_string(f"{result.id}[{i}]") for i in range(16)]


@_decorators.codegen(load_bfloat16_x16_to_float16, "cute")
def _(state: CodegenState) -> list[ast.AST]:
    storage = state.proxy_arg(0)
    if not isinstance(storage, torch.Tensor):
        raise TypeError("load_bfloat16_x16_to_float16 storage must be a tensor")

    base = state.device_function.tensor_arg(storage).name
    group_offset = state.ast_arg(1)
    extra_mask = state.ast_args[2]
    qwords = []
    for word in range(4):
        load = expr_from_string(
            f"cute.arch.load({base}.iterator + ({{offset}} * 16 + {word * 4}), "
            "cutlass.Uint64)",
            offset=group_offset,
        )
        if extra_mask is not None:
            assert isinstance(extra_mask, ast.AST)
            load = expr_from_string(
                "({load} if {mask} else cutlass.Uint64(0))",
                load=load,
                mask=extra_mask,
            )
        qwords.append(state.codegen.lift(load, dce=True, prefix="bf16_qword"))
    call = expr_from_string(
        "_cute_bfloat16_x16_to_float16({q0}, {q1}, {q2}, {q3})",
        q0=qwords[0],
        q1=qwords[1],
        q2=qwords[2],
        q3=qwords[3],
    )
    result = state.codegen.lift(call, dce=True, prefix="bf16_lanes")
    return [expr_from_string(f"{result.id}[{i}]") for i in range(16)]
