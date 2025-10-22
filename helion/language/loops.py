from __future__ import annotations

import ast
import builtins
import inspect
import itertools
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import TypeGuard
from typing import cast
from typing import overload

import torch
from torch._inductor.runtime.triton_heuristics import (
    get_max_y_grid,  # type: ignore[import-untyped]
)
import triton.language

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import warning
from .._compiler.type_propagation import GridIndexType
from .._compiler.type_propagation import IterType
from .._compiler.type_propagation import LiteralType
from .._compiler.type_propagation import Origin
from .._compiler.type_propagation import SequenceType
from .._compiler.type_propagation import TileIndexType
from .._compiler.type_propagation import TypeInfo
from .._compiler.variable_origin import GetItemOrigin
from ..autotuner.config_spec import ConfigSpec
from ..autotuner.config_spec import FlattenLoopSpec
from ..autotuner.config_spec import L2GroupingSpec
from ..autotuner.config_spec import LoopOrderSpec
from ..autotuner.config_spec import RangeFlattenSpec
from ..autotuner.config_spec import RangeMultiBufferSpec
from ..autotuner.config_spec import RangeNumStagesSpec
from ..autotuner.config_spec import RangeUnrollFactorSpec
from ..autotuner.config_spec import RangeWarpSpecializeSpec
from ..autotuner.config_spec import StaticRangeSpec
from . import _decorators
from .ref_tile import RefTile
from .tile_proxy import Tile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._compiler.inductor_lowering import CodegenState
    from .constexpr import ConstExpr


__all__ = ["grid", "static_range", "tile"]


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: int | torch.Tensor,
    end_or_none: int | torch.Tensor | None = None,
    /,
    block_size: object = None,
) -> Iterator[Tile]: ...


@overload
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: Sequence[int | torch.Tensor],
    end_or_none: Sequence[int | torch.Tensor] | None = None,
    /,
    block_size: object = None,
) -> Iterator[Sequence[Tile]]: ...


@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def tile(
    begin_or_end: int | torch.Tensor | Sequence[int | torch.Tensor],
    end_or_none: int | torch.Tensor | Sequence[int | torch.Tensor] | None = None,
    /,
    block_size: object = None,
) -> Iterator[Tile] | Iterator[Sequence[Tile]]:
    """
    Break up an iteration space defined by a size or sequence of sizes into tiles.

    The generated tiles can flatten the iteration space into the product of the sizes,
    perform multidimensional tiling, swizzle the indices for cache locality, reorder
    dimensions, etc. The only invariant is that every index in the range of the given
    sizes is covered exactly once.

    The exact tiling strategy is determined by a Config object, typically created
    through autotuning.

    If used at the top level of a function, this becomes the grid of the kernel.
    Otherwise, it becomes a loop in the output kernel.

    The key difference from :func:`~helion.language.grid` is that ``tile`` gives you
    ``Tile`` objects that load a slice of elements, while ``grid`` gives you scalar
    integer indices.  It is recommended to use ``tile`` in most cases, since it allows
    more choices in autotuning.

    Args:
        begin_or_end: If 2+ positional args provided, the start of iteration space.
                      Otherwise, the end of iteration space.
        end_or_none: If 2+ positional args provided, the end of iteration space.
        block_size: Fixed block size (overrides autotuning) or None for autotuned size

    Returns:
        Iterator[Tile] or Iterator[Sequence[Tile]]: Iterator over tile objects

    Examples:
        One dimensional tiling:

        .. code-block:: python

            @helion.kernel
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                result = torch.zeros_like(x)

                for tile in hl.tile(x.size(0)):
                    # tile processes multiple elements at once
                    result[tile] = x[tile] + y[tile]

                return result

        Multi-dimensional tiling:

        .. code-block:: python

            @helion.kernel()
            def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                m, k = x.size()
                k, n = y.size()
                out = torch.empty([m, n], dtype=x.dtype, device=x.device)

                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                    out[tile_m, tile_n] = acc


            return out

        Fixed block size:

        .. code-block:: python

            @helion.kernel
            def process_with_fixed_block(x: torch.Tensor) -> torch.Tensor:
                result = torch.zeros_like(x)

                for tile in hl.tile(x.size(0), block_size=64):
                    # Process with fixed block size of 64
                    result[tile] = x[tile] * 2

                return result

        Using tile properties:

        .. code-block:: python

            @helion.kernel
            def tile_info_example(x: torch.Tensor) -> torch.Tensor:
                result = torch.zeros([x.size(0)], dtype=x.dtype, device=x.device)

                for tile in hl.tile(x.size(0)):
                    # Access tile properties
                    start = tile.begin
                    end = tile.end
                    size = tile.block_size
                    indices = tile.index  # [start, start+1, ..., end-1]

                    # Use in computation
                    result[tile] = x[tile] + indices

                return result

    See Also:
        - :func:`~helion.language.grid`: For explicit control over the launch grid
        - :func:`~helion.language.tile_index`: For getting tile indices
        - :func:`~helion.language.register_block_size`: For registering block sizes

    Note:
        Similar to ``range()`` with multiple forms:

        * tile(end) iterates 0 to end-1, autotuned block_size
        * tile(begin, end) iterates begin to end-1, autotuned block_size
        * tile(begin, end, block_size) iterates begin to end-1, fixed block_size
        * tile(end, block_size=block_size) iterates 0 to end-1, fixed block_size

        Block sizes can be registered for autotuning explicitly with :func:`~helion.language.register_block_size`
        and passed as the ``block_size`` argument if one needs two loops to use the same block size.  Passing
        ``block_size=None`` is equivalent to calling register_block_size.

        Use ``tile`` in most cases. Use ``grid`` when you need explicit control over the launch grid.
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


def _allow_static_range(begin: object, end: object, step: object) -> bool:
    """
    Only enable tl.stagic_range when:
    1) The ranges are statically known at compile time.
    2) The range is small enough to be unrolled without blowing up the compile time.
    """
    if begin is None:
        begin = 0
    elif not isinstance(begin, int):
        return False

    if not isinstance(end, int):
        return False

    if step is None:
        count = end - begin
    elif isinstance(step, int):
        # Use integer math to avoid Triton constexpr types
        # Compute ceil((end - begin) / step)
        delta = end - begin
        count = (delta + step - 1) // step
    else:
        return False
    # Unrolling a long static range leads to compile timeouts
    return count <= 8


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
            raise exc.TypeInferenceError(
                f"expected IntLike or list[IntLike], got {begin_or_end!s}"
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
        proxy_block_size = Tile._tiles_to_sizes(_to_proxy(block_size))
        _check_matching(proxy_end, proxy_block_size)
    else:
        proxy_block_size = begin.tree_map(lambda n: None)

    if unpack := not isinstance(proxy_end, (list, tuple)):
        begin_list: list[int | torch.SymInt | torch.Tensor] = [
            cast("int | torch.SymInt | torch.Tensor", proxy_begin)
        ]
        end_list: list[int | torch.SymInt | torch.Tensor] = [
            cast("int | torch.SymInt | torch.Tensor", proxy_end)
        ]
        block_size_list: list[int | torch.SymInt | torch.Tensor | None] = [
            cast("int | torch.SymInt | torch.Tensor | None", proxy_block_size)
        ]
    else:
        begin_list = cast("list[int | torch.SymInt | torch.Tensor]", proxy_begin)
        end_list = cast("list[int | torch.SymInt | torch.Tensor]", proxy_end)
        block_size_list = cast(
            "list[int | torch.SymInt | torch.Tensor | None]", proxy_block_size
        )
    block_size_list = Tile._tiles_to_sizes(block_size_list)

    if unpack:
        target = getattr(parent, "target", None)
        if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) > 1:
            raise exc.FailedToUnpackTile from None

    results = []
    for begin_part, end_part, bs in zip(
        begin_list,
        end_list,
        block_size_list,
        strict=True,
    ):
        if isinstance(begin_part, Tile) or isinstance(end_part, Tile):
            raise exc.TileOfTile
        size = end_part - begin_part  # type: ignore[operator]
        if isinstance(size, int) and size < 0:
            raise exc.InvalidTileRange(begin_part, end_part)
        if isinstance(size, torch.Tensor):
            size = None  # data dependent size
        if bs is None:
            results.append(TileIndexType.allocate(size, origin))
        elif isinstance(bs, int):
            results.append(TileIndexType.allocate(size, origin, bs))
        elif isinstance(bs, torch.SymInt):
            from .._compiler.compile_environment import CompileEnvironment

            index = CompileEnvironment.current().get_block_id(bs)
            if index is None:
                results.append(TileIndexType.allocate(size, origin, bs))
            else:
                results.append(TileIndexType(origin=origin, block_id=index))
                CompileEnvironment.current().block_sizes[index].mark_alternate_size(
                    size
                )

    _add_config_choices(
        [x.block_id for x in results],
        is_tile=True,
        has_begin=not all((isinstance(x, int) and x == 0) for x in begin_list),
        allow_static_ranges=[
            *starmap(
                _allow_static_range,
                zip(begin_list, end_list, block_size_list, strict=True),
            )
        ],
    )
    if unpack:
        (result,) = results
    else:
        result = SequenceType(origin, tuple(results))
    return IterType(origin, result)


def _add_config_choices(
    block_ids: list[int],
    *,
    is_tile: bool = False,
    has_begin: bool = False,
    allow_static_ranges: list[bool] | None = None,
) -> None:
    config_spec = CompileEnvironment.current().config_spec

    if len(block_ids) > 1:
        # Add loop reordering choice
        config_spec.loop_orders.append(LoopOrderSpec(block_ids))
        if is_tile and not has_begin:
            config_spec.flatten_loops.append(FlattenLoopSpec(block_ids))

    is_grid = all(x._loop_type != LoopType.GRID for x in ExtendedAST.current())
    if is_grid:
        # Track which block_ids come from grids
        existing_ids = {*config_spec.grid_block_ids}
        config_spec.grid_block_ids.extend(
            [x for x in block_ids if x not in existing_ids]
        )
        if len(block_ids) >= 2:
            # L2 grouping now supports 3D+ grids by applying to innermost 2 dimensions
            config_spec.l2_groupings.append(L2GroupingSpec(block_ids))
        if not _allow_use_yz_grid(config_spec, block_ids):
            config_spec.disallow_pid_type("xyz")
        # just one set of choices for when we have persistent kernel loop
        _add_config_range_choice(block_ids)
    else:
        if allow_static_ranges is None:
            allow_static_ranges = [False] * len(block_ids)
        for block_id, allow_static_range in zip(
            block_ids, allow_static_ranges, strict=True
        ):
            _add_config_range_choice([block_id], allow_static_range=allow_static_range)


def _add_config_range_choice(
    block_ids: list[int], allow_static_range: bool = False
) -> None:
    params = inspect.signature(triton.language.range).parameters
    config_spec = CompileEnvironment.current().config_spec
    if allow_static_range:
        config_spec.static_ranges.append(StaticRangeSpec(block_ids))
    if "loop_unroll_factor" in params:
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec(block_ids))
    if _supports_warp_specialize() and "warp_specialize" in params:
        config_spec.range_warp_specialize.append(RangeWarpSpecializeSpec(block_ids))
    if "num_stages" in params:
        config_spec.range_num_stages.append(RangeNumStagesSpec(block_ids))
    if "disallow_acc_multi_buffer" in params:
        config_spec.range_multi_buffers.append(RangeMultiBufferSpec(block_ids))
    if "flatten" in params:
        config_spec.range_flattens.append(RangeFlattenSpec(block_ids))


def _supports_warp_specialize() -> bool:
    """Check if the current device supports warp specialization."""
    env = CompileEnvironment.current()
    if env.device.type != "cuda" or not env.settings.allow_warp_specialize:
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


def _allow_use_yz_grid(config_spec: ConfigSpec, block_ids: list[int]) -> bool:
    """Check if the yz grid is allowed based on the block sizes."""
    if not (1 < len(block_ids) <= 3):
        return False
    hint = 1
    try:
        for block_id in block_ids:
            hint *= config_spec.block_sizes.block_id_lookup(block_id).size_hint
    except KeyError:
        return False
    return hint < get_max_y_grid()


@_decorators.codegen(tile)
def _(state: CodegenState) -> ast.AST:
    return _codegen_loop_helper(state)


def _to_int(value: int | torch.Tensor | None) -> int | None:
    """Convert tensor values to int."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _normalize_to_list(
    value: int | torch.Tensor | list[int | torch.Tensor],
) -> list[int | torch.Tensor]:
    """Convert single values to lists for uniform handling."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_begin_end_ref(
    begin_or_end: int | torch.Tensor | list[int | torch.Tensor],
    end_or_none: int | torch.Tensor | list[int | torch.Tensor] | None = None,
) -> tuple[
    int | torch.Tensor | list[int | torch.Tensor],
    int | torch.Tensor | list[int | torch.Tensor],
]:
    if end_or_none is not None:
        # Two positional args: begin_or_end is begin, end_or_none is end
        return begin_or_end, end_or_none
    # One positional arg: begin_or_end is end, begin defaults to 0
    end = begin_or_end
    if isinstance(end, (list, tuple)):
        begin = cast("int | torch.Tensor | list[int | torch.Tensor]", [0] * len(end))
    else:
        begin = 0
    return begin, end


@_decorators.ref(tile)
def _(
    begin_or_end: int | torch.Tensor | list[int | torch.Tensor],
    end_or_none: int | torch.Tensor | list[int | torch.Tensor] | None = None,
    block_size: int | torch.Tensor | list[int | torch.Tensor] | None = None,
) -> Iterator[RefTile | tuple[RefTile, ...]]:
    # Issue warning if block_size is specified in interpret mode
    if block_size is not None:
        warning(exc.BlockSizeIgnoredInInterpretMode(block_size))

    # Step 1: Normalize begin and end values
    begin, end = _normalize_begin_end_ref(begin_or_end, end_or_none)

    # Step 2: Convert to lists and then to ints
    begin_list = _normalize_to_list(begin)
    end_list = _normalize_to_list(end)
    begin_ints = [_to_int(b) for b in begin_list]
    end_ints = [_to_int(e) for e in end_list]

    # Step 3: Determine block sizes - always return full dimension size, ignoring block_size parameter
    block_size_list = []
    for b, e in zip(begin_ints, end_ints, strict=True):
        assert b is not None and e is not None
        block_size_list.append(e - b)

    # Step 4: Determine return type
    # Return single tiles if input was not a list
    return_single = not isinstance(begin, list) and not isinstance(end, list)

    # Step 5: Generate tiles
    # Build tiles for each dimension
    tiles = []
    for b, e in zip(begin_ints, end_ints, strict=True):
        assert b is not None and e is not None
        if b != e:
            # Only create tile if range is non-empty
            tiles.append(RefTile(b, e, e - b))

    # Yield result based on return type
    if tiles:  # Only yield if we have at least one non-empty dimension
        if return_single:
            # Single dimension case - yield the tile directly
            assert len(tiles) == 1
            yield tiles[0]
        else:
            # Multi-dimensional case - yield as tuple
            yield tuple(tiles)


def _codegen_loop_helper(
    state: CodegenState,
) -> ast.AST:
    """Helper method for codegen of tile and grid decorators."""
    for_loop = ExtendedAST.current()[-2]
    loop_type = for_loop._loop_type
    type_info = ExtendedAST.current()[-1]._type_info
    assert isinstance(for_loop, ast.For)
    assert isinstance(type_info, IterType)

    if isinstance(type_info.inner, SequenceType):
        indices_raw = type_info.inner.unpack()
    else:
        indices_raw = [type_info.inner]
    assert all(isinstance(t, (TileIndexType, GridIndexType)) for t in indices_raw)
    indices = cast("list[TileIndexType | GridIndexType]", indices_raw)

    if loop_type == LoopType.GRID:
        env = CompileEnvironment.current()
        env.loop_dependency_checker.register_loop(for_loop)
        block_ids = [t.block_id for t in indices]
        state.tile_strategy.codegen_grid(state, block_ids)
        return expr_from_string("None")
    raise AssertionError(f"Expected loop type: {loop_type}")


@overload
@_decorators.device_func_replacement(builtins.range)
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    begin_or_end: int | torch.Tensor | ConstExpr,
    end_or_none: int | torch.Tensor | ConstExpr | None = None,
    /,
    step: int
    | torch.Tensor
    | ConstExpr
    | Sequence[int | torch.Tensor | ConstExpr]
    | None = None,
) -> Iterator[torch.SymInt]: ...


@overload
@_decorators.device_func_replacement(builtins.range)
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    begin_or_end: Sequence[int | torch.Tensor | ConstExpr],
    end_or_none: Sequence[int | torch.Tensor | ConstExpr] | None = None,
    /,
    step: int
    | torch.Tensor
    | ConstExpr
    | Sequence[int | torch.Tensor | ConstExpr]
    | None = None,
) -> Iterator[Sequence[torch.SymInt]]: ...


@_decorators.device_func_replacement(builtins.range)
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    begin_or_end: int
    | torch.Tensor
    | ConstExpr
    | Sequence[int | torch.Tensor | ConstExpr],
    end_or_none: int
    | torch.Tensor
    | ConstExpr
    | Sequence[int | torch.Tensor | ConstExpr]
    | None = None,
    /,
    step: int
    | torch.Tensor
    | ConstExpr
    | Sequence[int | torch.Tensor | ConstExpr]
    | None = None,
) -> Iterator[torch.SymInt] | Iterator[Sequence[torch.SymInt]]:  # type: ignore[type-arg]
    """Iterate over individual indices of the given iteration space.

    The key difference from :func:`~helion.language.tile` is that ``grid`` gives you
    scalar integer indices (``torch.SymInt``), while ``tile`` gives you ``Tile`` objects
    that load a slice of elements. Use ``tile`` in most cases. Use ``grid`` when you need
    explicit control over the launch grid or when processing one element at a time.

    Semantics are equivalent to:

    .. code-block:: python

        for i in hl.tile(...):
            # i is a Tile object, accesses multiple elements
            data = tensor[i]  # loads slice of elements (1D tensor)

    vs:

    .. code-block:: python

        for i in hl.grid(...):
            # i is a scalar index, accesses single element
            data = tensor[i]  # loads single element (0D scalar)

    When used at the top level of a function, this becomes the grid of the kernel.
    Otherwise, it becomes a loop in the output kernel.

    Args:
        begin_or_end: If 2+ positional args provided, the start of iteration space.
                      Otherwise, the end of iteration space.
        end_or_none: If 2+ positional args provided, the end of iteration space.
        step: Step size for iteration (default: 1)

    Returns:
        Iterator[torch.SymInt] or Iterator[Sequence[torch.SymInt]]: Iterator over scalar indices

    See Also:
        - :func:`~helion.language.tile`: For processing multiple elements at once
        - :func:`~helion.language.tile_index`: For getting tile indices
        - :func:`~helion.language.arange`: For creating index sequences

    Note:
        Similar to ``range()`` with multiple forms:

        * grid(end) iterates from 0 to end-1, step 1
        * grid(begin, end) iterates from begin to end-1, step 1
        * grid(begin, end, step) iterates from begin to end-1, given step
        * grid(end, step=step) iterates from 0 to end-1, given step

        Use ``tile`` in most cases. Use ``grid`` when you need explicit control over the launch grid.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(grid)
def _(
    begin_or_end: TypeInfo,
    end_or_none: TypeInfo | None = None,
    /,
    step: TypeInfo | None = None,
    *,
    origin: Origin,
) -> TypeInfo:
    parent = ExtendedAST.current()[-2]
    if not isinstance(parent, ast.For):
        raise exc.LoopFunctionNotInFor("grid")
    begin, end = _normalize_begin_end(begin_or_end, end_or_none, origin=origin)
    proxy_begin = _to_proxy(begin)
    proxy_end = _to_proxy(end)
    _check_matching(proxy_begin, proxy_end)
    if _not_none(step):
        proxy_step = Tile._tiles_to_sizes(_to_proxy(step))
        _check_matching(proxy_end, proxy_step)
    else:
        proxy_step = begin.tree_map(lambda n: None)

    if unpack := not isinstance(proxy_end, (list, tuple)):
        begin_list: list[int | torch.SymInt | torch.Tensor] = [
            cast("int | torch.SymInt | torch.Tensor", proxy_begin)
        ]
        end_list: list[int | torch.SymInt | torch.Tensor] = [
            cast("int | torch.SymInt | torch.Tensor", proxy_end)
        ]
        step_list: list[int | torch.SymInt | torch.Tensor | None] = [
            cast("int | torch.SymInt | torch.Tensor | None", proxy_step)
        ]
    else:
        begin_list = cast("list[int | torch.SymInt | torch.Tensor]", proxy_begin)
        end_list = cast("list[int | torch.SymInt | torch.Tensor]", proxy_end)
        step_list = cast("list[int | torch.SymInt | torch.Tensor | None]", proxy_step)

    results = []
    for begin_part, end_part, step_part in zip(
        begin_list,
        end_list,
        step_list,
        strict=True,
    ):
        size = end_part - begin_part  # type: ignore[operator]
        if isinstance(size, torch.Tensor):
            size = None  # data dependent size
        if step_part is None:
            step_part = 1
        results.append(GridIndexType.allocate(size, origin, step_part))  # pyright: ignore[reportArgumentType]

    _add_config_choices(
        [x.block_id for x in results],
        is_tile=False,
        has_begin=not all((isinstance(x, int) and x == 0) for x in begin_list),
        allow_static_ranges=[
            *starmap(
                _allow_static_range, zip(begin_list, end_list, step_list, strict=True)
            )
        ],
    )
    if unpack:
        (result,) = results
    else:
        result = SequenceType(origin, tuple(results))
    return IterType(origin, result)


@_decorators.codegen(grid)
def _(state: CodegenState) -> ast.AST:
    return _codegen_loop_helper(state)


def _extract_step_value(
    step: int | torch.Tensor | Sequence[int | torch.Tensor] | None,
    index: int = 0,
) -> int | torch.Tensor | None:
    """Extract step value from various input formats."""
    if step is None:
        return None

    if isinstance(step, (list, tuple)):
        # Extract from sequence at index
        if index < len(step):
            val = step[index]
            # Type narrow to valid types for _to_int
            if isinstance(val, (int, torch.Tensor, type(None))):
                return val
        return None

    # Single value - type narrow to valid types
    if isinstance(step, (int, torch.Tensor)):
        return step
    return None


def _normalize_step_values(
    step: int | torch.Tensor | Sequence[int | torch.Tensor] | None,
    num_dims: int,
) -> list[int | None]:
    """Normalize step values to a list of ints for each dimension."""
    if step is None:
        return [None] * num_dims

    assert isinstance(step, (list, tuple))
    step_ints = []
    for i in range(num_dims):
        step_val = _extract_step_value(step, i)
        step_ints.append(_to_int(step_val))
    return step_ints


def _create_ranges(
    begin_ints: list[int | None],
    end_ints: list[int | None],
    step_ints: list[int | None] | None = None,
) -> list[range]:
    """Create range objects from begin, end, and optional step values."""
    ranges = []

    if step_ints is None:
        # No steps provided - use default ranges
        for b, e in zip(begin_ints, end_ints, strict=True):
            assert b is not None and e is not None
            ranges.append(range(b, e))
    else:
        # Steps provided - use them where available
        for b, e, s in zip(begin_ints, end_ints, step_ints, strict=True):
            assert b is not None and e is not None
            if s is not None:
                ranges.append(range(b, e, s))
            else:
                ranges.append(range(b, e))

    return ranges


@_decorators.ref(grid)
def _(
    begin_or_end: int | torch.Tensor | list[int | torch.Tensor],
    end_or_none: int | torch.Tensor | list[int | torch.Tensor] | None = None,
    step: int | torch.Tensor | Sequence[int | torch.Tensor] | None = None,
) -> range | Iterator[tuple[int, ...]]:
    # Step 1: Normalize begin and end values
    begin, end = _normalize_begin_end_ref(begin_or_end, end_or_none)

    # Step 2: Handle single dimension case
    if not isinstance(begin, (list, tuple)):
        begin_int = _to_int(begin)
        assert not isinstance(end, (list, tuple))
        end_int = _to_int(end)
        assert begin_int is not None and end_int is not None

        # Extract step for single dimension
        step_val = _extract_step_value(step, 0)
        step_int = _to_int(step_val)

        if step_int is not None:
            return range(begin_int, end_int, step_int)
        return range(begin_int, end_int)

    # Step 3: Handle multi-dimensional case
    assert isinstance(end, (list, tuple))
    begin_ints = [_to_int(b) for b in begin]
    end_ints = [_to_int(e) for e in end]

    # Step 4: Normalize step values
    step_ints = (
        _normalize_step_values(step, len(begin_ints)) if step is not None else None
    )

    # Step 5: Create ranges and return product
    ranges = _create_ranges(begin_ints, end_ints, step_ints)
    return itertools.product(*ranges)


@_decorators.device_func_replacement(builtins.zip)
@_decorators.api(is_device_only=True, cache_type=True)
def _zip_replacement(
    *args: tuple[object, ...] | list[object],
    strict: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """
    Device replacement for zip() that returns tuples for unrolling.

    This replacement enables zip() to work in device kernels by converting
    the zip result to a tuple of tuples, which can then be unrolled by the
    existing tuple iteration logic.

    Args:
        *args: Sequences to zip together

    Returns:
        Tuple of tuples containing zipped elements

    Examples:
        .. code-block:: python

            @helion.kernel
            def kernel_with_zip(a_tensors, b_tensors):
                for a, b in zip(a_tensors, b_tensors):
                    # This gets unrolled at compile time
                    result += a * b
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(_zip_replacement)
def _(
    *args: TypeInfo,
    origin: Origin,
    **kwargs: object,
) -> TypeInfo:
    """Type propagation for zip replacement that preserves tensor types."""
    # Accept but ignore the strict keyword argument
    if not args:
        return SequenceType(origin, ())

    # Convert all arguments to SequenceType
    sequences = []
    for arg in args:
        if not isinstance(arg, SequenceType):
            raise exc.TypeInferenceError(
                f"zip() argument must be a sequence, got {arg}"
            )
        sequences.append(arg.unpack())

    # Check all sequences have the same length
    length = 0
    if sequences:
        length = len(sequences[0])
        for i, seq in enumerate(sequences[1:], 1):
            if len(seq) != length:
                raise exc.TypeInferenceError(
                    f"zip() argument {i} has length {len(seq)}, expected {length}"
                )

    # Build result as tuple of tuples, preserving existing TypeInfo objects
    result_elements = []
    for i in range(length):
        # Create a tuple containing the i-th element from each sequence
        tuple_elements = tuple(seq[i] for seq in sequences)
        tuple_type = SequenceType(GetItemOrigin(origin, i), tuple_elements)
        result_elements.append(tuple_type)

    return SequenceType(origin, tuple(result_elements))


@_decorators.register_to_device_ir(_zip_replacement)
def _(
    tracer: object,
    *flat_args: object,
) -> object:
    """Device IR handler for zip - returns the zipped result for unrolling."""
    # flat_args contains the prepared arguments: (tensor_sequences, strict_value)
    if not flat_args:
        return ()

    # Extract sequences and strict parameter
    if len(flat_args) == 2:
        sequences = flat_args[0]  # This should be the tuple of sequences
        strict = flat_args[1]  # This should be the strict parameter
        assert isinstance(strict, bool)
    else:
        assert len(flat_args) == 1
        sequences = flat_args[0]
        strict = False
    return [*builtins.zip(*sequences, strict=strict)]  # type: ignore[arg-type]


@_decorators.device_func_replacement(builtins.enumerate)
@_decorators.api(is_device_only=True, cache_type=True)
def _enumerate_replacement(
    iterable: tuple[object, ...] | list[object],
    start: int = 0,
) -> tuple[tuple[int, object], ...]:
    """
    Device replacement for enumerate() that returns tuples for unrolling.

    This replacement enables enumerate() to work in device kernels by converting
    the enumerate result to a tuple of (index, value) tuples, which can then be
    unrolled by the existing tuple iteration logic.

    Args:
        iterable: Sequence to enumerate
        start: Starting value for the counter (default: 0)

    Returns:
        Tuple of (index, value) tuples
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(_enumerate_replacement)
def _(
    iterable: TypeInfo,
    start: TypeInfo | None = None,
    *,
    origin: Origin,
) -> TypeInfo:
    """Type propagation for enumerate replacement that preserves tensor types."""
    if not isinstance(iterable, SequenceType):
        raise exc.TypeInferenceError(
            f"enumerate() argument must be a sequence, got {iterable}"
        )

    # Get the start value
    start_value = 0
    if start is not None and start.is_literal():
        start_val = start.as_literal()
        if isinstance(start_val, int):
            start_value = start_val

    # Build result as tuple of (index, value) tuples
    sequence_elements = iterable.unpack()
    result_elements = []

    for i, element in enumerate(sequence_elements):
        # Create (index, value) tuple
        index_literal = LiteralType(origin, start_value + i)
        tuple_elements = (index_literal, element)
        tuple_type = SequenceType(GetItemOrigin(origin, i), tuple_elements)
        result_elements.append(tuple_type)

    return SequenceType(origin, tuple(result_elements))


@_decorators.register_to_device_ir(_enumerate_replacement)
def _(
    tracer: object,
    *flat_args: object,
) -> object:
    """Device IR handler for enumerate - returns the enumerated result for unrolling."""
    if len(flat_args) == 2:
        iterable = flat_args[0]
        start = flat_args[1]
        assert isinstance(start, int)
    else:
        assert len(flat_args) == 1
        iterable = flat_args[0]
        start = 0
    return [*builtins.enumerate(iterable, start=start)]  # type: ignore[arg-type]


@_decorators.api(is_device_only=True, cache_type=True)
def static_range(
    begin_or_end: int,
    end_or_none: int | None = None,
    /,
    step: int = 1,
) -> Iterator[int]:
    """
    Create a range that gets unrolled at compile time by iterating over constant integer values.

    This function is similar to Python's built-in range(), but it generates a sequence
    of integer constants that triggers loop unrolling behavior in Helion kernels. The loop
    is completely unrolled at compile time, with each iteration becoming separate
    instructions in the generated code.

    Args:
        begin_or_end: If 2+ positional args provided, the start of range (integer).
                      Otherwise, the end of range (integer).
        end_or_none: If 2+ positional args provided, the end of range (integer).
        step: Step size for iteration (integer, default: 1)

    Returns:
        Iterator[int]: Iterator over constant integer values

    Examples:
        Simple unrolled loop:

        .. code-block:: python

            @helion.kernel
            def unrolled_example(x: torch.Tensor) -> torch.Tensor:
                result = torch.zeros_like(x)

                for tile in hl.tile(x.size(0)):
                    acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
                    # This loop gets completely unrolled
                    for i in hl.static_range(3):
                        acc += x[tile] * i
                    result[tile] = acc

                return result

        Range with start and step:

        .. code-block:: python

            @helion.kernel
            def kernel_stepped_unroll(x: torch.Tensor) -> torch.Tensor:
                result = torch.zeros_like(x)

                for tile in hl.tile(x.size(0)):
                    acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
                    # Unroll loop from 2 to 8 with step 2: [2, 4, 6]
                    for i in hl.static_range(2, 8, 2):
                        acc += x[tile] * i
                    result[tile] = acc

                return result

    Note:
        - Only constant integer values are supported
        - The range must be small enough to avoid compilation timeouts
        - Each iteration becomes separate instructions in the generated Triton code
        - Use for small, fixed iteration counts where unrolling is beneficial
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(static_range)
def _(
    begin_or_end: int,
    end_or_none: int | None = None,
    /,
    step: int = 1,
) -> tuple[int, ...]:
    """Fake function for static_range - validates integer constants and returns tuple(range(...))."""
    # Validate that inputs are compile-time constants
    if end_or_none is not None:
        begin_val = begin_or_end
        end_val = end_or_none
    else:
        begin_val = 0
        end_val = begin_or_end

    if (
        not isinstance(begin_val, int)
        or not isinstance(end_val, int)
        or not isinstance(step, int)
    ):
        raise exc.TypeInferenceError("static_range requires constant integer arguments")

    # Return tuple(range(...)) which will trigger existing tuple/list unrolling
    return tuple(range(begin_val, end_val, step))


@_decorators.ref(static_range)
def _(
    begin_or_end: int,
    end_or_none: int | None = None,
    step: int = 1,
) -> range:
    if end_or_none is not None:
        return range(begin_or_end, end_or_none, step)
    return range(begin_or_end)
