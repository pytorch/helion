from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["float4_e2m1fn_x2_to_float32"]


@_decorators.api(is_device_only=True)
def float4_e2m1fn_x2_to_float32(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a ``torch.float4_e2m1fn_x2`` scalar tensor into FP32 lanes."""
    raise exc.NotInsideKernel


@_decorators.register_fake(float4_e2m1fn_x2_to_float32)
def _(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dtype is not torch.float4_e2m1fn_x2:
        raise exc.InvalidAPIUsage(
            "hl.float4_e2m1fn_x2_to_float32 expects a "
            f"torch.float4_e2m1fn_x2 tensor, got {value.dtype}"
        )
    return (
        torch.empty(value.shape, dtype=torch.float32, device=value.device),
        torch.empty(value.shape, dtype=torch.float32, device=value.device),
    )


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
