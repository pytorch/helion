from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import TypeGuard
from typing import overload

import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.tile_index_proxy import TileIndexProxy
from .._compiler.type_propagation import GridIndexType
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

__all__ = ["grid", "register_block_size", "tile"]


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: int,
    end_or_none: int | None = None,
    /,
    block_size: object = None,
) -> Iterator[TileOutput]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: Sequence[int],
    end_or_none: Sequence[int] | None = None,
    /,
    block_size: object = None,
) -> Iterator[Sequence[TileOutput]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: int | Sequence[int],
    end_or_none: int | Sequence[int] | None = None,
    /,
    block_size: object = None,
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

    Similar to `range()` there are multiple forms of this function:
        tile(end) iterates from 0 to `end - 1`, with autotuned block_size.
        tile(begin, end) iterates from `begin` to `end - 1`, with autotuned block_size.
        tile(begin, end, block_size) iterates from `begin` to `end - 1`, with the given block_size.
        tile(end, block_size=block_size) iterates from 0 to `end - 1`, with the given block_size.

    begin/end/block_size can be a single integer or a sequence of integers to specify
    multidimensional iteration.  Block sizes can be explicitly registered for autotuning
    with `hl.register_block_size()`.

    Examples:

        for tile in hl.tile(1000):
            ...

        for tile0, tile1 in hl.tile([1000, 1000]):
            ...

    :param begin_or_end: If 2 or more positional arguments are provided, the start of the iteration space.  Otherwise, the end of the iteration space.
    :param end_or_none: If 2 or more positional arguments are provided, the end of the iteration space.
    :return: A TileIndexProtocol object if a single size is provided, or a sequence of TileIndexProtocol objects if a sequence of sizes is provided.
    """
    raise exc.NotInsideKernel


def _not_none(value: TypeInfo | None) -> TypeGuard[TypeInfo]:
    return not (value is None or value.is_literal() and value.as_literal() is None)


def _to_proxy(value: TypeInfo) -> object:
    try:
        return value.proxy()
    except NotImplementedError:
        raise exc.IncorrectTileUsage(
            f"expected IntLike or list[IntLike], got {value!s}"
        ) from None


def _check_matching(a: object, b: object) -> None:
    """Check that the types of `a` and `b` match for use in hl.tile."""
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            raise exc.IncorrectTileUsage(
                f"expected type hl.tile args to match, got {type(a)} and {type(b)}"
            )
        if len(a) != len(b):
            raise exc.IncorrectTileUsage(
                f"expected dims for hl.tile args to match, got {len(a)} and {len(b)}"
            )
    elif isinstance(a, (int, torch.SymInt, torch.Tensor)):
        if not isinstance(b, (int, torch.SymInt, torch.Tensor)):
            raise exc.IncorrectTileUsage(
                f"expected type hl.tile args to match, got {type(a)} and {type(b)}"
            )
    else:
        raise exc.IncorrectTileUsage(
            f"expected type hl.tile args to be IntLike or list[IntLike], got {type(a)}"
        )


def _normalize_begin_end(
    begin_or_end: TypeInfo,
    end_or_none: TypeInfo | None,
    origin: Origin,
) -> tuple[TypeInfo, TypeInfo]:
    """Fill in defaults for begin if it is not provided."""
    if _not_none(end_or_none):
        begin = begin_or_end
        end = end_or_none
    else:
        try:
            begin = TypeInfo.from_example(begin_or_end.tree_map(lambda n: 0), origin)
        except NotImplementedError:
            raise exc.TypePropagationError(
                UnknownType(
                    origin,
                    f"expected IntLike or list[IntLike], got {begin_or_end!s}",
                    chained_from=begin_or_end,
                )
            ) from None
        end = begin_or_end
    return begin, end


@_decorators.type_propagation(tile)
def _(
    begin_or_end: TypeInfo,
    end_or_none: TypeInfo | None = None,
    /,
    block_size: TypeInfo | None = None,
    *,
    origin: Origin,
) -> TypeInfo:
    parent = ExtendedAST.current()[-2]
    if not isinstance(parent, ast.For):
        raise exc.LoopFunctionNotInFor("tile")
    begin, end = _normalize_begin_end(begin_or_end, end_or_none, origin=origin)
    proxy_begin = _to_proxy(begin)
    proxy_end = _to_proxy(end)
    _check_matching(proxy_begin, proxy_end)
    if _not_none(block_size):
        proxy_block_size = TileIndexProxy.tiles_to_sizes(_to_proxy(block_size))
        _check_matching(proxy_end, proxy_block_size)
    else:
        proxy_block_size = begin.tree_map(lambda n: None)

    if unpack := not isinstance(proxy_end, (list, tuple)):
        proxy_begin = [proxy_begin]
        proxy_end = [proxy_end]
        proxy_block_size = [proxy_block_size]

    if (
        all(bs is None for bs in proxy_block_size)
        and all(isinstance(s, (int, torch.SymInt)) for s in proxy_begin)
        and all(isinstance(s, (int, torch.SymInt)) for s in proxy_end)
    ):
        proxy_size = [e - b for b, e in zip(proxy_begin, proxy_end, strict=True)]
        results = TileIndexType.allocate(proxy_size, origin)
    else:
        # we must allocate the block sizes individually due to data dependent size or pre-allocated block sizes
        # TODO(jansel): this flattens the structure of the config, which we should avoid
        results = []
        for begin_part, end_part, bs in zip(
            proxy_begin, proxy_end, proxy_block_size, strict=True
        ):
            size = end_part - begin_part
            if isinstance(size, torch.Tensor):
                size = None  # data dependent size
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
                    CompileEnvironment.current().block_sizes[index].mark_alternate_size(
                        size
                    )
    if unpack:
        (result,) = results
    else:
        result = SequenceType(origin, results)
    return IterType(origin, result)


def _get_block_indices(type_info: TypeInfo) -> list[int]:
    def visit(n: TypeInfo) -> TypeInfo:
        if isinstance(n, (TileIndexType, GridIndexType)):
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
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(sizes: int, /) -> Iterator[torch.SymInt]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(sizes: Sequence[int], /) -> Iterator[Sequence[torch.SymInt]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    sizes: int | Sequence[int],
    /,
) -> Iterator[torch.SymInt] | Iterator[Sequence[torch.SymInt]]:  # type: ignore[type-arg]
    """Iterate over *individual* indices of the given iteration space.

    Semantics are equivalent to

        for i in hl.tile(size, block_size=1):
            ...

    but `i` will be a scalar (`torch.SymInt`), not a 1-element tensor.
    """

    raise exc.NotInsideKernel


@_decorators.type_propagation(grid)
def _(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    parent = ExtendedAST.current()[-2]
    if not isinstance(parent, ast.For):
        raise exc.LoopFunctionNotInFor("grid")
    try:
        proxy_sizes = sizes.proxy()
        if not (
            isinstance(proxy_sizes, (int, torch.SymInt))
            or (
                isinstance(proxy_sizes, (list, tuple))
                and all(isinstance(x, (int, torch.SymInt)) for x in proxy_sizes)
            )
        ):
            raise NotImplementedError
    except NotImplementedError:
        raise exc.TypePropagationError(
            UnknownType(
                origin,
                f"grid() expected int or list[int], got {sizes!s}",
                chained_from=sizes,
            )
        ) from None

    if isinstance(proxy_sizes, (int, torch.SymInt)):
        return IterType(origin, GridIndexType.allocate(proxy_sizes, origin))

    assert isinstance(proxy_sizes, (list, tuple))
    elements = [GridIndexType.allocate(s, origin) for s in proxy_sizes]
    return IterType(origin, SequenceType(origin, elements))


@_decorators.codegen(grid)
def _(state: CodegenState) -> ast.AST:
    for_loop = ExtendedAST.current()[-2]
    loop_type = for_loop._loop_type
    type_info = ExtendedAST.current()[-1]._type_info
    assert isinstance(for_loop, ast.For)
    assert isinstance(type_info, IterType)
    if isinstance(type_info.inner, SequenceType):
        grid_indices = type_info.inner.unpack()
    else:
        grid_indices = [type_info.inner]
    assert all(isinstance(t, GridIndexType) for t in grid_indices)
    if loop_type == LoopType.GRID:
        block_indices = [t.block_size_idx for t in grid_indices]
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


def _register_block_size_types(sizes: TypeInfo, origin: Origin) -> TypeInfo:
    proxy_sizes = sizes.proxy()
    if isinstance(proxy_sizes, (int, torch.SymInt)):
        return TileIndexType.allocate([proxy_sizes], origin)[0]
    return SequenceType(
        origin=origin,
        # pyre-fixme[6]
        element_types=TileIndexType.allocate(proxy_sizes, origin),
    )


@_decorators.type_propagation(register_block_size)
def _(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _register_block_size_types(sizes, origin)
