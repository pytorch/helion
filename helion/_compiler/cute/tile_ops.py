"""CuTe-backend codegen for the tile ops defined in ``helion.language.tile_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``tile_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.tile_ops import _disable_flatten_get_tile
from ...language.tile_ops import tile_begin
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(tile_begin, "cute")
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0), state)
    from ..tile_strategy import NDTileStrategy

    loops = state.codegen.active_device_loops.get(index)
    grid_state = state.codegen.current_grid_state
    strategy = (
        loops[-1].strategy
        if loops
        else (grid_state.strategy if grid_state is not None else None)
    )
    if isinstance(strategy, NDTileStrategy) and index in strategy.block_ids:
        # _grid_local_coord_expr is index - offset for an ND tile, including
        # its blocked/strided/vector lanes and any cluster CTA slice. Use the
        # authoritative logical offset directly instead of index-(index-offset).
        # Besides removing redundant integer arithmetic, this keeps a uniform
        # tile boundary from creating false lane-index dependencies in siblings.
        return expr_from_string(strategy.offset_var(index))

    global_index = state.codegen.index_var(index)

    thread_axis = None
    if loops:
        from .cute_reshape import _per_thread_nd_tile_offset

        # Use the same innermost owner as GenerateAST.index_var, not a
        # possibly different outer/current grid in a nested loop.
        tile_offset = _per_thread_nd_tile_offset(loops[-1].strategy, index)
        if tile_offset is not None:
            return expr_from_string(tile_offset)
        thread_axis = loops[-1].block_thread_axes.get(index)
    if thread_axis is None:
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(index)
    if thread_axis is None:
        return expr_from_string(state.codegen.offset_var(index))

    from .cute_reshape import _grid_local_coord_expr

    local_coord = _grid_local_coord_expr(state.codegen, index, thread_axis)
    return state.codegen.lift(
        expr_from_string(f"({global_index}) - ({local_coord})"),
        dce=True,
        prefix="tile_begin",
    )
