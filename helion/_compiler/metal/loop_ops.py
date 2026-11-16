"""Metal-backend codegen for ops defined in ``helion.language.loops``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.loops import _codegen_loop_helper
from ...language.loops import tile

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
