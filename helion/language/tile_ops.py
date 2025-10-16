from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.host_function import HostFunction
from .._compiler.host_function import SymbolOrigin
from .._compiler.variable_origin import GridOrigin
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .ref_tile import RefTile
    from .tile_interface import TileInterface


def _register_tile_symbol_origin(symbol: torch.SymInt, tile_index: int) -> None:
    """Register the origin for a tile-related symbol so it can be resolved during codegen."""
    HostFunction.current().expr_to_origin[symbol._sympy_()] = SymbolOrigin(
        GridOrigin(tile_index)
    )


@_decorators.api(tiles_as_sizes=True)
def tile_index(tile: TileInterface) -> torch.Tensor:
    """
    Retrieve the index (a 1D tensor containing offsets) of the given tile.
    This can also be written as: `tile.index`.

    Example usage::

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
    env = CompileEnvironment.current()
    assert env.get_block_id(tile) is not None
    return torch.empty([tile], dtype=env.settings.index_dtype, device=env.device)


@_decorators.codegen(tile_index)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.index_var(index))


@_decorators.ref(tile_index)
def _(tile: RefTile) -> torch.Tensor:
    env = CompileEnvironment.current()
    return torch.arange(
        tile._slice.start, tile._slice.stop, dtype=torch.int32, device=env.device
    )


@_decorators.api(tiles_as_sizes=True)
def tile_begin(tile: TileInterface) -> int:
    """
    Retrieve the start offset of the given tile.
    This can also be written as: `tile.begin`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_begin)
def _(tile: torch.SymInt) -> torch.SymInt:
    index = _disable_flatten_get_tile(tile)  # update config spec if needed
    result = CompileEnvironment.current().cached_create_unbacked_symint(
        ("tile_begin", tile)
    )
    _register_tile_symbol_origin(result, index)
    return result


def _disable_flatten_get_tile(tile: object) -> int:
    """Helper to extract tile index from state."""
    assert isinstance(tile, torch.SymInt), (type(tile), tile)
    env = CompileEnvironment.current()
    index = env.get_block_id(tile)
    assert index is not None
    # The functions in this file can't be used in flattened loops.
    env.config_spec.flatten_loops.disable_block_id(index)
    return index


@_decorators.codegen(tile_begin)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.offset_var(index))


@_decorators.ref(tile_begin)
def _(tile: RefTile) -> int:
    return tile._slice.start


@_decorators.api(tiles_as_sizes=True)
def tile_end(tile: TileInterface) -> int:
    """
    Retrieve the end offset of the given tile.
    For the first 0 to N-1 tiles, this is equivalent to `tile.begin + tile.block_size`.
    For the last tile, this is the end offset passed to `hl.tile()`.
    This can also be written as: `tile.end`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_end)
def _(tile: torch.SymInt) -> torch.SymInt:
    index = _disable_flatten_get_tile(tile)  # update config spec if needed
    result = CompileEnvironment.current().cached_create_unbacked_symint(
        ("tile_end", tile)
    )
    _register_tile_symbol_origin(result, index)
    return result


@_decorators.codegen(tile_end)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    offset_var = state.codegen.offset_var(index)
    block_size_var = state.device_function.block_size_var(index)
    if block_size_var is None:
        block_size_var = "1"
    naive_exp = f"{offset_var} + {block_size_var}"
    if state.codegen.mask_var(index) is not None:
        # if masking is used, we must update the end bound of the last tile
        end_var = (
            state.codegen.active_device_loops[index][-1]
            .block_id_to_info[index]
            .end_var_name
        )
        return expr_from_string(f"tl.minimum({naive_exp}, {end_var})")
    # If we don't have a mask, we can simply return the offset + block size
    return expr_from_string(naive_exp)


@_decorators.ref(tile_end)
def _(tile: RefTile) -> int:
    return tile._slice.stop


@_decorators.api(tiles_as_sizes=True)
def tile_block_size(tile: TileInterface) -> int:
    """
    Retrieve block size of a given tile, usually set the autotuner.
    This can also be written as: `tile.block_size`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_block_size)
def _(tile: torch.SymInt) -> torch.SymInt:
    return tile


# since we return tile above, no codegen is needed for this function.
# codegen is handled in _get_symnode()


@_decorators.ref(tile_block_size)
def _(tile: RefTile) -> int:
    return tile._block_size


@_decorators.api(tiles_as_sizes=True)
def tile_count(tile: TileInterface) -> int:
    """
    Retrieve the number of tiles along the given tile dimension.
    This is equivalent to ``cdiv(tile_end, tile.block_size)`` when iterating
    from 0, and more generally ``cdiv(end - begin, block_size)`` for an
    iteration space [begin, end).

    This can also be written as: `tile.count`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_count)
def _(tile: torch.SymInt) -> torch.SymInt:
    index = _disable_flatten_get_tile(tile)
    result = CompileEnvironment.current().cached_create_unbacked_symint(
        ("tile_count", tile)
    )
    _register_tile_symbol_origin(result, index)
    return result


@_decorators.codegen(tile_count)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    # Use device loop metadata to get end and block size
    end_var = (
        state.codegen.active_device_loops[index][-1]
        .block_id_to_info[index]
        .end_var_name
    )
    block_size_var = state.device_function.block_size_var(index)
    if block_size_var is None:
        block_size_var = "1"
    return expr_from_string(f"tl.cdiv({end_var}, {block_size_var})")


@_decorators.ref(tile_count)
def _(tile: RefTile) -> int:
    # Number of tiles covering [begin, end) at granularity block_size
    begin = tile._slice.start
    end = tile._slice.stop
    bs = tile._block_size
    return (end - begin + bs - 1) // bs


@_decorators.api(tiles_as_sizes=True)
def tile_id(tile: TileInterface) -> int:
    """
    Retrieve tile_id of a given tile or list of tiles.
    This is equivalent to `tile.begin // tile.block_size`.
    This can also be written as: `tile.id`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_id)
def _(tile: torch.SymInt) -> torch.SymInt:
    index = _disable_flatten_get_tile(tile)  # update config spec if needed
    assert isinstance(tile, torch.SymInt)
    result = CompileEnvironment.current().cached_create_unbacked_symint(
        ("tile_id", tile)
    )
    _register_tile_symbol_origin(result, index)
    return result


@_decorators.codegen(tile_id)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    offset = state.codegen.offset_var(index)
    block_size = state.device_function.block_size_var(index)
    if block_size is None:
        expr_str = offset
    else:
        expr_str = f"{offset} // {block_size}"
    return expr_from_string(expr_str)


@_decorators.ref(tile_id)
def _(tile: RefTile) -> int:
    # ID is always 0 since we always have one tile per dim in ref mode
    return 0
