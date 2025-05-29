from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.tile_strategy import TileStrategy
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .loops import Tile


@_decorators.api(tiles_as_sizes=True)
def tile_index(tile: Tile) -> torch.Tensor:
    """
    Retrieve the index (a 1D tensor containing offsets) of the given tile.
    This can also be written as: `tile.index`.

    Example usage:

        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty(length, dtype=torch.int32, device=device)
            for tile in hl.tile(length):
               out[tile] = tile.index
            return out
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_index)
def _(tile: torch.SymInt) -> torch.Tensor:
    assert isinstance(tile, torch.SymInt)
    assert TileStrategy.get_block_index(tile) is not None
    env = CompileEnvironment.current()
    return torch.empty([tile], dtype=env.settings.index_dtype, device=env.device)


@_decorators.codegen(tile_index)
def _(state: CodegenState) -> ast.AST:
    tile = state.proxy_arg(0)
    assert isinstance(tile, torch.SymInt)
    index = TileStrategy.get_block_index(tile)
    assert index is not None
    return expr_from_string(state.codegen.index_var(index))
