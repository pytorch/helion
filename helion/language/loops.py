from __future__ import annotations

import ast
import builtins
import inspect
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import TypeGuard
from typing import overload

import torch
from torch._inductor.runtime.triton_heuristics import get_max_y_grid
import triton.language

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import LoopType
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.type_propagation import GridIndexType
from .._compiler.type_propagation import IterType
from .._compiler.type_propagation import Origin
from .._compiler.type_propagation import SequenceType
from .._compiler.type_propagation import TileIndexType
from .._compiler.type_propagation import TypeInfo
from ..autotuner.config_spec import ConfigSpec
from ..autotuner.config_spec import FlattenLoopSpec
from ..autotuner.config_spec import L2GroupingSpec
from ..autotuner.config_spec import LoopOrderSpec
from ..autotuner.config_spec import RangeFlattenSpec
from ..autotuner.config_spec import RangeMultiBufferSpec
from ..autotuner.config_spec import RangeNumStagesSpec
from ..autotuner.config_spec import RangeUnrollFactorSpec
from ..autotuner.config_spec import RangeWarpSpecializeSpec
from . import _decorators
from helion.language.tile_proxy import Tile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._compiler.inductor_lowering import CodegenState


__all__ = ["grid", "tile"]


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
        proxy_begin = [proxy_begin]
        proxy_end = [proxy_end]
        proxy_block_size = [proxy_block_size]

    results = []
    for begin_part, end_part, bs in zip(
        proxy_begin, proxy_end, proxy_block_size, strict=True
    ):
        size = end_part - begin_part
        if isinstance(size, torch.Tensor):
            size = None  # data dependent size
        if bs is None:
            results.append(TileIndexType.allocate(size, origin))
        elif isinstance(bs, int):
            results.append(TileIndexType.allocate(size, origin, bs))
        elif isinstance(bs, torch.SymInt):
            from helion._compiler.compile_environment import CompileEnvironment

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
        has_begin=not all((isinstance(x, int) and x == 0) for x in proxy_begin),
    )
    if unpack:
        (result,) = results
    else:
        result = SequenceType(origin, results)
    return IterType(origin, result)


def _add_config_choices(
    block_ids: list[int], *, is_tile: bool = False, has_begin: bool = False
) -> None:
    config_spec = CompileEnvironment.current().config_spec

    if len(block_ids) > 1:
        # Add loop reordering choice
        config_spec.loop_orders.append(LoopOrderSpec(block_ids))
        if is_tile and not has_begin:
            config_spec.flatten_loops.append(FlattenLoopSpec(block_ids))

    is_grid = all(x._loop_type != LoopType.GRID for x in ExtendedAST.current())
    if is_grid:
        if len(block_ids) == 2:
            # TODO(jansel): support L2 grouping with 3+ dims (and maybe non-grids?)
            config_spec.l2_groupings.append(L2GroupingSpec(block_ids))
        if not _allow_use_yz_grid(config_spec, block_ids):
            config_spec.disallow_pid_type("xyz")
    else:
        params = inspect.signature(triton.language.range).parameters
        for block_id in block_ids:
            if "loop_unroll_factor" in params:
                config_spec.range_unroll_factors.append(
                    RangeUnrollFactorSpec([block_id])
                )
            if _supports_warp_specialize() and "warp_specialize" in params:
                config_spec.range_warp_specialize.append(
                    RangeWarpSpecializeSpec([block_id])
                )
            if "num_stages" in params:
                config_spec.range_num_stages.append(RangeNumStagesSpec([block_id]))
            if "disallow_acc_multi_buffer" in params:
                config_spec.range_multi_buffers.append(RangeMultiBufferSpec([block_id]))
            if "flatten" in params:
                config_spec.range_flattens.append(RangeFlattenSpec([block_id]))


def _supports_warp_specialize() -> bool:
    """Check if the current device supports warp specialization."""
    env = CompileEnvironment.current()
    if env.device.type != "cuda" or not env.settings.allow_warp_specialize:
        return False
    return torch.cuda.get_device_capability() >= (12, 0)


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
        indices = type_info.inner.unpack()
    else:
        indices = [type_info.inner]
    assert all(isinstance(t, (TileIndexType, GridIndexType)) for t in indices)

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
    begin_or_end: int | torch.Tensor,
    end_or_none: int | torch.Tensor | None = None,
    /,
    step: object = None,
) -> Iterator[torch.SymInt]: ...


@overload
@_decorators.device_func_replacement(builtins.range)
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    begin_or_end: Sequence[int | torch.Tensor],
    end_or_none: Sequence[int | torch.Tensor] | None = None,
    /,
    step: object = None,
) -> Iterator[Sequence[torch.SymInt]]: ...


@_decorators.device_func_replacement(builtins.range)
@_decorators.api(
    is_device_loop=True, is_device_only=False, cache_type=True, tiles_as_sizes=True
)
def grid(
    begin_or_end: int | torch.Tensor | Sequence[int | torch.Tensor],
    end_or_none: int | torch.Tensor | Sequence[int | torch.Tensor] | None = None,
    /,
    step: object = None,
) -> Iterator[torch.SymInt] | Iterator[Sequence[torch.SymInt]]:  # type: ignore[type-arg]
    """Iterate over individual indices of the given iteration space.

    Semantics are equivalent to

        for i in hl.tile(...):
            ...

    but `i` will be a scalar (`torch.SymInt`), not a 1-element tensor.

    When used at the top level of a function, this becomes the grid of the kernel.
    Otherwise, it becomes a loop in the output kernel.

    Similar to `range()` there are multiple forms of this function:
        grid(end) iterates from 0 to `end - 1`, with step size 1.
        grid(begin, end) iterates from `begin` to `end - 1`, with step size 1.
        grid(begin, end, step) iterates from `begin` to `end - 1`, with the given step size.
        grid(end, step=step) iterates from 0 to `end - 1`, with the given step size.
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
        proxy_begin = [proxy_begin]
        proxy_end = [proxy_end]
        proxy_step = [proxy_step]

    results = []
    for begin_part, end_part, step_part in zip(
        proxy_begin, proxy_end, proxy_step, strict=True
    ):
        size = end_part - begin_part
        if isinstance(size, torch.Tensor):
            size = None  # data dependent size
        if step_part is None:
            step_part = 1
        results.append(GridIndexType.allocate(size, origin, step_part))

    _add_config_choices(
        [x.block_id for x in results],
        is_tile=False,
        has_begin=not all((isinstance(x, int) and x == 0) for x in proxy_begin),
    )
    if unpack:
        (result,) = results
    else:
        result = SequenceType(origin, results)
    return IterType(origin, result)


@_decorators.codegen(grid)
def _(state: CodegenState) -> ast.AST:
    return _codegen_loop_helper(state)
