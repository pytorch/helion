"""Triton-backend codegen for ops defined in ``helion.language._tracing_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``_tracing_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.utils import triton_type

from ...language import _decorators
from ...language._tracing_ops import _mask_to
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_mask_to, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_exprs: list[str] = []
    input_sizes = [*tensor.size()]
    env = CompileEnvironment.current()
    for dim, size in enumerate(input_sizes):
        if (index := env.resolve_block_id(size)) is not None and (
            mask_var := state.codegen.mask_var(index)
        ) is not None:
            expand = state.tile_strategy.expand_str(input_sizes, dim)
            if env.is_jagged_tile(index):
                mask_shape = env.jagged_tile_mask_shapes[index]
                expand = state.tile_strategy.jagged_tile_expand_str(
                    mask_shape, input_sizes
                )

            expr = f"({mask_var}{expand})"
            if expr not in mask_exprs:
                mask_exprs.append(expr)
    if not mask_exprs:
        return state.ast_arg(0)
    mask_expr = "&".join(mask_exprs)
    if len(mask_exprs) < len(input_sizes):
        mask_expr = f"tl.broadcast_to({mask_expr}, {state.tile_strategy.shape_str(input_sizes)})"
    # Ensure the masked value literal matches the tensor dtype to avoid unintended upcasts
    input_dtype = tensor.dtype
    other_typed = expr_from_string(
        f"tl.full([], {constant_repr(other)}, {triton_type(input_dtype)})"
    )
    return expr_from_string(
        f"tl.where({mask_expr}, {{expr}}, {{other}})",
        expr=state.ast_arg(0),
        other=other_typed,
    )
