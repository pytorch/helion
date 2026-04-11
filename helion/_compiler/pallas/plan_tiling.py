"""Tiling analysis pass for the Pallas backend.

Analyzes indexing expressions to determine which tensor dimensions can be tiled.
Sets 'dim_tilings' metadata on tensors based on indexing constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import sympy
import torch

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..host_function import SymbolOrigin
    from ..tile_dispatch import TileStrategyDispatch


@dataclass
class IndexingPattern:
    """Base class for indexing patterns detected during tiling analysis."""


@dataclass
class TilePattern(IndexingPattern):
    """Vanilla tile pattern - translates to ':' when tiled."""

    block_id: int


@dataclass
class TileIndexWithOffsetPattern(IndexingPattern):
    """Tile index with offset - no tiling allowed."""

    block_id: int
    offset: int | torch.SymInt | object


@dataclass
class TileBeginWithOffsetPattern(IndexingPattern):
    """Tile begin with offset - allow/disallow tiling based on bounds."""

    block_id: int
    offset: int | torch.SymInt | object


@dataclass
class ArbitrarySlicePattern(IndexingPattern):
    slice: slice


@dataclass
class ArbitraryIndexPattern(IndexingPattern):
    index: int | torch.SymInt | object | None


@dataclass
class NonePattern(IndexingPattern):
    """None index pattern (broadcasting dimension) - allow tiling."""


@dataclass
class DimensionTiling:
    """Tiling decision for a specific dimension of a tensor

    can_tile: whether or not we can tile this dimension
    block_ids: which which block_ids we are indexing this dimension (there can be multiple, in which case we mustn't tile)
    """

    can_tile: bool = True
    block_ids: list[int] = field(default_factory=list)


def plan_tiling(
    graphs: list[GraphInfo],
    config: Config,
    tile_strategy: TileStrategyDispatch,
) -> None:
    for graph_info in graphs:
        _analyze_indexing_expressions(graph_info, config)


def _analyze_indexing_expressions(graph_info: GraphInfo, config: Config) -> None:
    from ...language import atomic_ops
    from ...language import memory_ops

    for node in graph_info.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in (
            memory_ops.load,
            memory_ops.store,
            atomic_ops.atomic_add,
            atomic_ops.atomic_cas,
            atomic_ops.atomic_or,
            atomic_ops.atomic_xor,
            atomic_ops.atomic_xchg,
            atomic_ops.atomic_min,
            atomic_ops.atomic_max,
            atomic_ops.atomic_and,
        ):
            _analyze_indexing(node, config)


def _analyze_indexing(node: torch.fx.Node, config: Config) -> None:
    tensor_arg = node.args[0]
    subscript = node.args[1]

    assert isinstance(subscript, (list, tuple))
    assert isinstance(tensor_arg, torch.fx.Node)
    tensor_val = tensor_arg.meta.get("val")
    assert isinstance(tensor_val, torch.Tensor)

    from helion._compiler.device_function import DeviceFunction

    device_fn = DeviceFunction.current()
    if id(tensor_val) not in device_fn.pallas_tensor_dim_tilings:
        device_fn.pallas_tensor_dim_tilings[id(tensor_val)] = [
            DimensionTiling() for _ in range(tensor_val.ndim)
        ]
    dim_tilings = device_fn.pallas_tensor_dim_tilings[id(tensor_val)]

    # Store indexing patterns directly on the memory operation node
    indexing_patterns = _analyze_subscript_patterns(
        tensor_val, list(subscript), dim_tilings, node, config
    )
    node.meta["indexing_patterns"] = indexing_patterns


def _analyze_subscript_patterns(
    tensor: torch.Tensor,
    subscript: list[object],
    dim_tilings: list[DimensionTiling],
    node: torch.fx.Node,
    config: Config,
) -> list[IndexingPattern]:
    """Analyze subscript patterns and create indexing pattern metadata."""
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    patterns: list[IndexingPattern] = []
    tensor_dim = 0  # Track which tensor dimension we're indexing

    for i, idx in enumerate(subscript):
        if idx is None:
            # None adds an unsqueezed dimension but doesn't consume a tensor dimension
            patterns.append(NonePattern())
            continue

        if tensor_dim >= tensor.ndim:
            raise AssertionError(
                f"Indexing {tensor_dim}th dim but tensor only has {tensor.ndim} dims"
            )

        # Detect different indexing patterns
        pattern = _detect_indexing_pattern(idx, tensor, tensor_dim, node, i, env)
        patterns.append(pattern)

        # Update dim_tilings based on the detected pattern
        _update_tiling_decision(tensor, pattern, tensor_dim, dim_tilings, env, config)

        tensor_dim += 1

    return patterns


def _detect_indexing_pattern(
    idx: object,
    tensor: torch.Tensor,
    tensor_dim: int,
    node: torch.fx.Node,
    subscript_index: int,
    env: CompileEnvironment,
) -> IndexingPattern:
    """Detect the specific indexing pattern for a subscript element."""
    from ..indexing_strategy import _get_tile_with_offset_info
    from ..variable_origin import GridOrigin

    if isinstance(idx, torch.fx.Node):
        idx_val = idx.meta.get("val")
        if isinstance(idx_val, torch.SymInt):
            block_id = env.get_block_id(idx_val)
            if block_id is not None:
                symbol_origin = _maybe_get_symbol_origin(idx_val)
                is_hl_grid = symbol_origin is not None and isinstance(
                    symbol_origin.origin, GridOrigin
                )
                if not is_hl_grid:
                    return TilePattern(block_id=block_id)

        tile_with_offset = _get_tile_with_offset_info(idx_val, node, subscript_index)
        if tile_with_offset is not None:
            return TileIndexWithOffsetPattern(
                block_id=tile_with_offset.block_id, offset=tile_with_offset.offset
            )

        # Check for TileBeginWithOffset pattern (t.begin, t.end-1)
        tile_begin_with_offset = _maybe_get_tile_begin_with_offset_info(idx_val)
        if tile_begin_with_offset is not None:
            return TileBeginWithOffsetPattern(
                block_id=tile_begin_with_offset.block_id,
                offset=tile_begin_with_offset.offset,
            )

    if isinstance(idx, slice):
        if idx != slice(None):
            raise AssertionError(
                f"Arbitrary slice expr {slice} not supported in Pallas backend yet"
            )
        return ArbitrarySlicePattern(idx)

    if isinstance(idx, (int, torch.SymInt)):
        return ArbitraryIndexPattern(idx)

    raise AssertionError(f"Unrecognized indexing pattern for pallas backend {idx}")


def _update_tiling_decision(
    tensor: torch.Tensor,
    pattern: IndexingPattern,
    tensor_dim: int,
    dim_tilings: list[DimensionTiling],
    env: CompileEnvironment,
    config: Config,
) -> None:
    """Update tiling decision based on the detected indexing pattern."""

    curr_dim_tiling = dim_tilings[tensor_dim]

    def _disallow_tiling() -> None:
        curr_dim_tiling.can_tile = False

    def _try_set_tiling_block_id(new_block_id: int) -> None:
        if new_block_id not in curr_dim_tiling.block_ids:
            curr_dim_tiling.block_ids.append(new_block_id)
            if len(curr_dim_tiling.block_ids) > 1:
                # we already need to tile this dim using a different block_id
                # so fallback to no-tiling so that we can access using both tiles
                _disallow_tiling()

    if isinstance(pattern, TilePattern):
        _try_set_tiling_block_id(pattern.block_id)

    elif isinstance(pattern, TileIndexWithOffsetPattern):
        _disallow_tiling()

    elif isinstance(pattern, TileBeginWithOffsetPattern):
        _try_set_tiling_block_id(pattern.block_id)
        # check bounds
        if not isinstance(pattern.offset, int) or pattern.offset < 0:
            _disallow_tiling()
        else:
            block_size = env.block_sizes[pattern.block_id].from_config(config)
            if isinstance(block_size, int) and pattern.offset >= block_size:
                _disallow_tiling()

    elif isinstance(pattern, ArbitrarySlicePattern):
        if pattern.slice != slice(None):
            # fow now we only support the `[:]` slice pattern
            _disallow_tiling()

    elif isinstance(pattern, ArbitraryIndexPattern):
        _disallow_tiling()

    elif isinstance(pattern, NonePattern):
        pass

    if isinstance(pattern, (TilePattern, TileBeginWithOffsetPattern)):
        block_size = env.block_sizes[pattern.block_id].from_config(config)
        if isinstance(block_size, int):
            from ..compile_environment import CompileEnvironment

            backend = CompileEnvironment.current().backend
            from helion._compiler.backend import PallasBackend

            assert isinstance(backend, PallasBackend)

            dim_from_end = tensor.ndim - tensor_dim - 1
            bitwidth = tensor.dtype.itemsize * 8
            required_alignment = backend._get_pallas_required_alignment(
                dim_from_end, tensor.ndim, bitwidth
            )

            if (
                block_size < tensor.shape[tensor_dim]
                and block_size % required_alignment != 0
            ):
                _disallow_tiling()


# Helper functions moved from memory_ops.py
def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    """Get symbol origin for a subscript element."""
    from ..compile_environment import _symint_expr
    from ..host_function import HostFunction

    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


def _maybe_get_tile_begin_with_offset_info(
    idx: object,
) -> TileBeginWithOffsetPattern | None:
    """Extended version that allows out-of-bounds and symbolic offsets."""
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import _symint_expr
    from ..host_function import HostFunction
    from ..host_function import SymbolOrigin
    from ..variable_origin import GridOrigin
    from ..variable_origin import TileBeginOrigin
    from ..variable_origin import TileEndOrigin

    idx_symbol_origin = _maybe_get_symbol_origin(idx)
    if isinstance(idx_symbol_origin, SymbolOrigin):
        if isinstance(idx_symbol_origin.origin, TileBeginOrigin):
            return TileBeginWithOffsetPattern(
                block_id=idx_symbol_origin.origin.block_id, offset=0
            )
        if isinstance(idx_symbol_origin.origin, GridOrigin) and not isinstance(
            idx_symbol_origin.origin, TileEndOrigin
        ):
            return TileBeginWithOffsetPattern(
                block_id=idx_symbol_origin.origin.block_id, offset=0
            )

    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if not isinstance(expr, sympy.Expr):
        return None

    args = expr.args
    origin: TileBeginOrigin | TileEndOrigin | GridOrigin | None = None
    offset = 0

    for arg in args:
        assert isinstance(arg, sympy.Expr)
        if (
            symbol_origin := HostFunction.current().expr_to_origin.get(arg)
        ) is not None:
            if isinstance(
                symbol_origin.origin, (GridOrigin, TileBeginOrigin, TileEndOrigin)
            ):
                if origin is not None:
                    # Multiple tile offset expressions - result is out of current tile
                    return None
                origin = symbol_origin.origin
            else:
                return None
        elif arg.is_constant():
            evalf_result = arg.evalf()
            f_value = float(evalf_result)  # type: ignore[arg-type]
            if not f_value.is_integer():
                return None
            offset += int(f_value)
        else:
            offset = torch.SymInt(arg)
            break

    env = CompileEnvironment.current()
    if origin is None:
        return None

    block_id = origin.block_id

    if isinstance(origin, TileEndOrigin):
        block_size = env.block_sizes[block_id].size
        if isinstance(block_size, int) and isinstance(offset, int):
            offset = block_size + offset  # Starting from end
        else:
            # For non-integer block sizes or offsets, fall back to symbolic offset
            offset = torch.SymInt(f"{block_size} + {offset}")  # type: ignore[arg-type]

    return TileBeginWithOffsetPattern(block_id=block_id, offset=offset)
