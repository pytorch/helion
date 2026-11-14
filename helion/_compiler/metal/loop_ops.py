"""Metal-backend codegen for ops defined in ``helion.language.loops`` and
``helion.language.tile_ops``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.loops import _codegen_loop_helper
from ...language.loops import tile
from ...language.tile_ops import _disable_flatten_get_tile
from ...language.tile_ops import tile_index
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(tile, "metal")
def _(state: CodegenState) -> ast.AST:
    # A root hl.tile's third argument is its block size, which the strategy
    # already holds.  The shared flattened strategy reads that argument as an
    # hl.grid step (FlattenedTileStrategy._extract_root_bounds), so a literal
    # or registered block size multiplied every index by itself; the N-D
    # strategy ignores it for a tile (_root_grid_steps).
    begin_or_end, end_or_none, _ = state.proxy_args
    return _codegen_loop_helper(
        state._replace(proxy_args=[begin_or_end, end_or_none, None])
    )


@_decorators.codegen(tile_index, "metal")
def _(state: CodegenState) -> ast.AST:
    # Metal keeps its loop indices as ``uint``, but tile.index is a tensor of
    # the signed index dtype.  Read the index variable as that dtype, or
    # arithmetic on it wraps: ``tile.index - 3 >= 0`` would be true for every
    # index.  Addresses keep using the unsigned index variable.
    index = _disable_flatten_get_tile(state.proxy_arg(0), state)
    env = CompileEnvironment.current()
    return expr_from_string(
        env.backend.cast_expr(
            state.codegen.index_var(index), env.backend.dtype_str(env.index_dtype)
        )
    )
