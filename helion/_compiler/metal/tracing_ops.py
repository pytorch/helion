"""Metal-backend codegen for ops defined in ``helion.language._tracing_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "metal")``
registrations; ``_tracing_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr

from ...language import _decorators
from ...language._tracing_ops import _mask_select
from ...language._tracing_ops import _mask_to
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast
    from collections.abc import Collection

    from ..inductor_lowering import CodegenState


def _tile_mask_expr(
    state: CodegenState,
    tensor: torch.Tensor,
    block_ids: Collection[int] | None = None,
) -> str | None:
    """The mask of the active tiles over ``tensor``'s dims (of the canonical
    ``block_ids`` only, if given), or None when no such dim has a masked
    tile."""
    env = CompileEnvironment.current()
    mask_exprs: list[str] = []
    for size in tensor.size():
        if (
            (index := env.resolve_block_id(size)) is not None
            and (block_ids is None or env.canonical_block_id(index) in block_ids)
            and (mask_var := state.codegen.mask_var(index)) is not None
            and mask_var not in mask_exprs
        ):
            mask_exprs.append(mask_var)
    return " and ".join(mask_exprs) if mask_exprs else None


@_decorators.codegen(_mask_to, "metal")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    other = state.proxy_arg(1)
    assert isinstance(other, (int, float, bool))
    mask_expr = _tile_mask_expr(state, tensor)
    if mask_expr is None:
        return state.ast_arg(0)
    input_dtype = tensor.dtype
    other_typed = CompileEnvironment.current().backend.cast_ast(
        expr_from_string(constant_repr(other)),
        input_dtype,
    )
    return expr_from_string(
        "({expr} if {mask} else {other})",
        expr=state.ast_arg(0),
        mask=expr_from_string(mask_expr),
        other=other_typed,
    )


@_decorators.codegen(_mask_select, "metal")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    block_ids = state.proxy_arg(2)
    assert isinstance(block_ids, tuple)
    mask_expr = _tile_mask_expr(state, tensor, block_ids)
    if mask_expr is None:
        return state.ast_arg(0)
    other = state.proxy_arg(1)
    return expr_from_string(
        "({expr} if {mask} else {other})",
        expr=state.ast_arg(0),
        mask=expr_from_string(mask_expr),
        other=CompileEnvironment.current().backend.cast_ast(
            expr_from_string(constant_repr(other)), tensor.dtype
        )
        if isinstance(other, (int, float, bool))
        else state.ast_arg(1),
    )
