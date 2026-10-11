"""CuTe-backend codegen for the tile ops defined in ``helion.language.tile_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``tile_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ... import exc
from ...language import _decorators
from ...language.tile_ops import _disable_flatten_get_tile
from ...language.tile_ops import tile_begin
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


def cute_tile_begin_expr(codegen: GenerateAST, index: int) -> str:
    """Uniform first index of the current tile along block ``index``.

    Shared by the ``tile.begin`` codegen and ``GenerateAST.tile_begin_var``
    (which also renders ``tile.end`` / ``tile.id`` and tile-edge symbols that
    survive into sympy expressions), so every edge derives from one begin.
    """
    from ..tile_strategy import NDTileStrategy
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from .cute_reshape import _grid_local_coord_expr
    from .cute_reshape import _per_thread_nd_tile_offset

    loops = codegen.active_device_loops.get(index)
    grid_state = codegen.current_grid_state
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
        return strategy.offset_var(index)
    if isinstance(strategy, PerThreadFlattenedTileStrategy):
        # The flattened per-thread tile records ``begin + pid * BLOCK``; its
        # thread-local coordinate is ``thread * elements_per_thread + lane``,
        # so ``index - thread_idx`` (the generic path below) is only the tile
        # base when each thread owns one element.
        base = strategy.cute_tile_base_expr(index)
        if base is not None:
            return base

    thread_axis = None
    if loops:
        # Use the same innermost owner as GenerateAST.index_var, not a
        # possibly different outer/current grid in a nested loop.
        tile_offset = _per_thread_nd_tile_offset(loops[-1].strategy, index)
        if tile_offset is not None:
            return tile_offset
        thread_axis = loops[-1].block_thread_axes.get(index)
    if thread_axis is None and grid_state is not None:
        thread_axis = grid_state.block_thread_axes.get(index)
    if thread_axis is None:
        # No thread axis owns this block: the strategy knows its tile base
        # (the loop offset, or ``begin + pid * BLOCK`` for the flattened per-thread
        # tile whose offset is already the per-element index).
        if strategy is not None:
            return strategy.tile_begin_var(index)
        return codegen.offset_var(index)

    global_index = codegen.index_var(index)
    local_coord = _grid_local_coord_expr(codegen, index, thread_axis)
    return ast.unparse(
        codegen.lift(
            expr_from_string(f"({global_index}) - ({local_coord})"),
            dce=True,
            prefix="tile_begin",
        )
    )


def cute_masked_block_end(codegen: GenerateAST, block_id: int, what: str) -> str | None:
    """The bound ``end`` that ``block_id``'s mask (``index < end``) checks.

    The mask is evaluated at this lane's own index; code that addresses other
    indices of the tile (a serial walk over it, a lane loop over a broadcast
    tile) compares them with ``end`` instead.  None when the block has no
    mask: every index of its tile is in range.
    """
    from ..tile_strategy import DeviceGridState

    if codegen.mask_var(block_id) is None:
        return None
    loops = codegen.active_device_loops.get(block_id)
    owner = loops[-1] if loops else codegen.current_grid_state
    info = owner.block_id_to_info.get(block_id) if owner is not None else None
    if CompileEnvironment.current().is_jagged_tile(block_id):
        raise exc.BackendUnsupported(
            "cute", f"{what} along a masked dim whose bound is not known"
        )
    if info is None:
        raise exc.BackendUnsupported(
            "cute", f"{what} along a masked dim without a logical tile owner"
        )
    if isinstance(owner, DeviceGridState):
        if info.grid_end_expr is None:
            raise exc.BackendUnsupported(
                "cute", f"{what} along a grid dim without a logical tile end"
            )
        return codegen.device_function.literal_expr(info.grid_end_expr)
    if info.end_var_name is not None:
        return info.end_var_name
    if info.end_expr is None:
        raise exc.BackendUnsupported(
            "cute", f"{what} along a dim with a data-dependent bound"
        )
    return codegen.device_function.sympy_expr(info.end_expr)


@_decorators.codegen(tile_begin, "cute")
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0), state)
    return expr_from_string(cute_tile_begin_expr(state.codegen, index))
