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
    global_index = state.codegen.index_var(index)

    thread_axis = None
    loops = state.codegen.active_device_loops.get(index)
    if loops:
        thread_axis = loops[-1].block_thread_axes.get(index)
    if thread_axis is None:
        grid_state = state.codegen.current_grid_state
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
