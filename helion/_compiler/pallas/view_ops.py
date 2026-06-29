"""Pallas-backend codegen for ops defined in ``helion.language.view_ops``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.view_ops import join
from ...language.view_ops import split
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from . import codegen as pallas_codegen

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(split, "pallas")
def _(state: CodegenState) -> list[ast.AST]:
    tensor = state.ast_arg(0)
    return [
        expr_from_string("{tensor}[..., 0]", tensor=tensor),
        expr_from_string("{tensor}[..., 1]", tensor=tensor),
    ]


@_decorators.codegen(join, "pallas")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "jnp.stack(jnp.broadcast_arrays({tensor0}, {tensor1}), axis=-1)",
        tensor0=state.ast_arg(0),
        tensor1=state.ast_arg(1),
    )


@_decorators.codegen(subscript, "pallas")
def codegen_subscript_pallas(state: CodegenState) -> ast.AST:
    output_keys: list[str] = []
    has_new_axis = False
    indices = state.proxy_arg(1)
    assert isinstance(indices, (list, tuple))
    for value in indices:
        if value is None:
            output_keys.append("None")
            has_new_axis = True
        elif isinstance(value, slice) and value == slice(None):
            output_keys.append(":")
        else:
            raise exc.InvalidIndexingType(repr(value))

    tensor = state.proxy_arg(0)
    if has_new_axis and isinstance(tensor, torch.Tensor) and tensor.dtype is torch.bool:
        output = state.fake_value
        assert isinstance(output, torch.Tensor)
        shape = state.tile_strategy.shape_str([*output.size()])
        return pallas_codegen.numeric_mask_reshape_expr(state.ast_arg(0), shape)

    return expr_from_string(
        f"{{base}}[{', '.join(output_keys)}]",
        base=state.ast_arg(0),
    )
