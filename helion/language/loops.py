from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Iterator
from typing import overload

import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.ast_extension import expr_from_string
from .._compiler.tile_index_proxy import TileIndexProxy
from .._compiler.type_propagation import IterType
from .._compiler.type_propagation import Origin
from .._compiler.type_propagation import SequenceType
from .._compiler.type_propagation import TileIndexType
from .._compiler.type_propagation import TypeInfo
from .._compiler.type_propagation import UnknownType
from . import _decorators

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._compiler.inductor_lowering import CodegenState

    # hl.tile doesn't actually return a tensor, but we say it does so user code can typecheck cleanly
    TileOutput = torch.Tensor

__all__ = ["register_block_size", "tile"]


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(sizes: int, block_size: TileOutput | None = None) -> Iterator[TileOutput]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    sizes: Sequence[int], block_size: Sequence[TileOutput] | None = None
) -> Iterator[Sequence[TileOutput]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    sizes: int | Sequence[int],
    block_size: TileOutput | Sequence[TileOutput] | None = None,
) -> Iterator[TileOutput] | Iterator[Sequence[TileOutput]]:
    """
    Break up an iteration space defined by a size or sequence of sizes into tiles.
    The generated tiles can flatten the iteration space into the product of the sizes,
    perform multidimensional tiling, swizzle the indices for cache locality, reorder
    dimensions, etc.  The only invariant is that every index in the range of the given
    sizes is covered exactly once.

    The exact tiling strategy is determined by a Config object, typically created
    through autotuning.

    If used at the top level of a function, this becomes the grid of the kernel.
    Otherwise, it becomes a loop in the output kernel.

    Examples:

        for tile in hl.tile(1000):
            ...

        for tile0, tile1 in hl.tile([1000, 1000]):
            ...

    :param sizes: An integer or a sequence of integers representing the sizes for tiling.
    :return: A TileIndexProtocol object if a single size is provided, or a sequence of TileIndexProtocol objects if a sequence of sizes is provided.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(tile)
def _(
    sizes: TypeInfo, block_size: TypeInfo | None = None, *, origin: Origin
) -> TypeInfo:
    parent = ExtendedAST.current()[-2]
    if not isinstance(parent, ast.For):
        raise exc.LoopFunctionNotInFor("tile")
    if (
        block_size is None
        or block_size.is_literal()
        and block_size.as_literal() is None
    ):
        result = _register_block_size_types(sizes, origin)
    else:
        try:
            proxy_sizes = sizes.proxy()
            proxy_block_size = TileIndexProxy.tiles_to_sizes(block_size.proxy())
        except NotImplementedError:
            raise exc.IncorrectTileUsage(
                f"expected int or list[int], got {sizes!s} and {block_size!s}"
            ) from None
        if isinstance(proxy_sizes, (list, tuple)):
            if not isinstance(proxy_block_size, (list, tuple)) or len(
                proxy_sizes
            ) != len(proxy_block_size):
                raise exc.IncorrectTileUsage(
                    f"expected dims for sizes and block_sizes to match, got {sizes!s} and {block_size!s}"
                )
            unpack = False
        else:
            if not isinstance(proxy_block_size, int | torch.SymInt):
                raise exc.IncorrectTileUsage(
                    f"expected type for sizes and block_sizes to match, got {sizes!s} and {block_size!s}"
                )
            proxy_sizes = [proxy_sizes]
            proxy_block_size = [proxy_block_size]
            unpack = True
        results = []
        for size, bs in zip(proxy_sizes, proxy_block_size, strict=True):
            if bs is None:
                results.append(TileIndexType.allocate([size], origin)[0])
            elif isinstance(bs, int):
                results.append(TileIndexType.allocate_fixed(size, bs, origin))
            elif isinstance(bs, torch.SymInt):
                from helion._compiler.tile_strategy import TileStrategy

                index = TileStrategy.get_block_index(bs)
                if index is None:
                    results.append(TileIndexType.allocate_fixed(size, bs, origin))
                else:
                    results.append(TileIndexType(origin=origin, block_size_idx=index))
        if unpack:
            (result,) = results
        else:
            result = SequenceType(origin, results)
    return IterType(origin, result)


def _register_block_size_types(sizes: TypeInfo, origin: Origin) -> TypeInfo:
    try:
        proxy_sizes = sizes.proxy()
        if not (
            isinstance(proxy_sizes, int | torch.SymInt)
            or isinstance(proxy_sizes, (tuple, list))
            and all(isinstance(x, (int, torch.SymInt)) for x in proxy_sizes)
        ):
            raise NotImplementedError
    except NotImplementedError:
        raise exc.TypePropagationError(
            UnknownType(
                origin,
                f"tile() expected int or list[int], got {sizes!s}",
                chained_from=sizes,
            )
        ) from None
    if isinstance(proxy_sizes, (int, torch.SymInt)):
        return TileIndexType.allocate([proxy_sizes], origin)[0]
    return SequenceType(
        origin=origin,
        # pyre-fixme[6]
        element_types=TileIndexType.allocate(proxy_sizes, origin),
    )


def _get_block_indices(type_info: TypeInfo) -> list[int]:
    def visit(n: TypeInfo) -> TypeInfo:
        if isinstance(n, TileIndexType):
            result.append(n.block_size_idx)
        return n

    result: list[int] = []
    type_info.tree_map(visit)
    return result


@_decorators.codegen(tile)
def _(state: CodegenState) -> ast.AST:
    for_loop = ExtendedAST.current()[-2]
    loop_type = for_loop._loop_type
    type_info = ExtendedAST.current()[-1]._type_info
    assert isinstance(for_loop, ast.For)
    assert isinstance(type_info, IterType)
    if isinstance(type_info.inner, SequenceType):
        tile_indices = type_info.inner.unpack()
    else:
        tile_indices = [type_info.inner]
    assert all(isinstance(t, TileIndexType) for t in tile_indices)
    if loop_type == LoopType.GRID:
        block_indices = [t.block_size_idx for t in tile_indices]
        state.tile_strategy.codegen_grid(state, block_indices)
        return expr_from_string("None")
    raise AssertionError(f"Expected loop type: {loop_type}")


@overload
@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_block_size(size: int) -> TileOutput: ...


@overload
@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_block_size(size: Sequence[int]) -> Sequence[TileOutput]: ...


@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_block_size(size: int | Sequence[int]) -> TileOutput | Sequence[TileOutput]:
    """
    Explicitly register a block size that should be autotuned and can
    be used for allocations and inside hl.tile().

    This is useful if you have two loops where you want them to share
    a block size, or if you need to allocate a kernel tensor before the
    hl.tile() loop.

    :param size:
    :return:
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(register_block_size)
def _(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _register_block_size_types(sizes, origin)
