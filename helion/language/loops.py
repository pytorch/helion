from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import overload

import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.ast_extension import expr_from_string
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

__all__ = ["grid", "register_block_size", "register_reduction_dim", "tile"]


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(sizes: int, block_size: object = None) -> Iterator[TileOutput]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    sizes: Sequence[int], block_size: object = None
) -> Iterator[Sequence[TileOutput]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    sizes: int | Sequence[int],
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
def grid(sizes: int) -> Iterator[torch.SymInt]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(sizes: Sequence[int]) -> Iterator[Sequence[torch.SymInt]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    sizes: int | Sequence[int],
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


@_decorators.type_propagation(register_block_size)
def _(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _register_block_size_types(sizes, origin)


@overload
@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_reduction_dim(size: int) -> torch.SymInt: ...


@overload
@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_reduction_dim(size: Sequence[int]) -> Sequence[torch.SymInt]: ...


@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_reduction_dim(size: int | Sequence[int]) -> torch.SymInt | Sequence[torch.SymInt]:
    """
    Explicitly register a reduction dimension that should be used for reduction operations.
    
    This is useful when you need to allocate a dimension for reduction that isn't
    automatically inferred from a slice operation. The registered dimension can be
    used for allocations and operations that require knowing the reduction size upfront.
    
    Unlike regular block sizes, reduction dimensions are specifically marked for
    reduction operations and may have different optimization strategies applied.
    
    :param size: An integer or a sequence of integers representing the reduction dimension sizes.
    :return: A SymInt object if a single size is provided, or a sequence of SymInt objects if a sequence of sizes is provided.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(register_reduction_dim)
def _(size: int | Sequence[int]) -> torch.SymInt | Sequence[torch.SymInt]:
    """Fake implementation that returns the registered reduction dimension size(s)"""
    from .._compiler.compile_environment import CompileEnvironment
    
    env = CompileEnvironment.current()
    
    if isinstance(size, (int, torch.SymInt)):
        # Allocate a single reduction dimension
        rdim = env.allocate_reduction_dimension(size)
        # Return the RDIM variable
        return rdim.var
    
    # Allocate multiple reduction dimensions
    results = []
    for s in size:
        rdim = env.allocate_reduction_dimension(s)
        # Return the RDIM variables
        results.append(rdim.var)
    
    return results


@_decorators.type_propagation(register_reduction_dim)
def _(sizes: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import ReductionDimType
    
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
                f"register_reduction_dim() expected int or list[int], got {sizes!s}",
                chained_from=sizes,
            )
        ) from None
    
    env = CompileEnvironment.current()
    
    # For single size
    if not isinstance(sizes, SequenceType):
        # Allocate a reduction dimension
        rdim = env.allocate_reduction_dimension(proxy_sizes)
        # Return a ReductionDimType that will use the RDIM variable
        return ReductionDimType(origin, rdim.block_size_idx)
    
    # For multiple sizes - return a sequence of ReductionDimTypes
    results = []
    for i, size in enumerate(proxy_sizes):
        rdim = env.allocate_reduction_dimension(size)
        results.append(ReductionDimType(origin, rdim.block_size_idx))
    
    return SequenceType(origin=origin, element_types=results)


@_decorators.codegen(register_reduction_dim)
def _(state: CodegenState) -> ast.AST:
    """Generate code for register_reduction_dim - return the size expression"""
    from .._compiler.type_propagation import ReductionDimType
    
    # In the generated host code, we need to return the actual size expression
    # The RDIM variable allocation already happened during type propagation
    
    # The type info should be a ReductionDimType or SequenceType of ReductionDimTypes
    current_node = ExtendedAST.current()[-1]
    type_info = current_node._type_info
    
    if isinstance(type_info, ReductionDimType):
        # Single reduction dim - we need to reconstruct the original size expression
        # from the AST node arguments
        if current_node.args:
            # Return the first argument which should be the size expression
            return current_node.args[0]
        else:
            # Fallback - this shouldn't happen
            raise NotImplementedError("No args found for register_reduction_dim")
    else:
        # Multiple reduction dims
        raise NotImplementedError("Multiple reduction dims not yet supported in codegen")
