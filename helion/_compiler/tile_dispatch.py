from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .device_ir import ForLoopGraphInfo
from .device_ir import ReductionLoopGraphInfo
from .host_function import HostFunction
from .reduction_strategy import LoopedReductionStrategy
from .reduction_strategy import PersistentReductionStrategy
from .reduction_strategy import ReductionStrategy
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceLoopState
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .. import Config
    from .inductor_lowering import CodegenState

    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


class TileStrategyDispatch:
    def __init__(
        self,
        fn: DeviceFunction,
        config: Config,
    ) -> None:
        super().__init__()
        self.strategies: list[TileStrategy] = []
        self.block_id_to_strategy: dict[tuple[int, ...], TileStrategy] = {}
        self._add_loop_strategies(fn, config)
        self._add_reduction_strategies(fn, config)

    def _add_loop_strategies(self, fn: DeviceFunction, config: Config) -> None:
        device_ir = HostFunction.current().device_ir
        for block_ids in device_ir.grid_block_ids:
            self._add_loop_strategy(block_ids, fn, config)
        for graph in device_ir.graphs:
            if isinstance(graph, ForLoopGraphInfo) and not isinstance(
                graph, ReductionLoopGraphInfo
            ):
                block_ids = [*graph.block_ids]
                self._add_loop_strategy(block_ids, fn, config)

    def _add_loop_strategy(
        self, block_ids: list[int], fn: DeviceFunction, config: Config
    ) -> None:
        env = CompileEnvironment.current()
        strategy = env.backend.create_loop_strategy(fn, block_ids, config)
        self.strategies.append(strategy)
        self.block_id_to_strategy[tuple(block_ids)] = strategy

    def _add_reduction_strategies(self, fn: DeviceFunction, config: Config) -> None:
        env = CompileEnvironment.current()
        rdims = [bs.block_id for bs in env.block_sizes if bs.reduction]
        for block_id in rdims:
            reduction_loop = env.config_spec.reduction_loops.config_get(
                config.reduction_loops, block_id, None
            )
            if reduction_loop is None:
                strategy: TileStrategy = PersistentReductionStrategy(fn, block_id)
            else:
                strategy = LoopedReductionStrategy(fn, block_id, reduction_loop)
            self.strategies.append(strategy)
            self.block_id_to_strategy[(block_id,)] = strategy

    def codegen_grid(self, state: CodegenState, block_ids: list[int]) -> None:
        strategy = self.block_id_to_strategy[tuple(block_ids)]
        grid_state = strategy.codegen_grid(state)
        for other_strategy in self.strategies:
            if other_strategy is not strategy:
                other_strategy.codegen_preamble(state)
        state.codegen.set_active_loops(grid_state)

    def codegen_device_loop(
        self, state: CodegenState, block_ids: list[int]
    ) -> DeviceLoopState:
        strategy = self.block_id_to_strategy[tuple(block_ids)]
        return strategy.codegen_device_loop(state)

    def _compact_shape(self, shapes: ShapeLike) -> list[CompactedShape]:
        compacted_shapes = []
        for idx, shape in enumerate(shapes):
            block_idx = CompileEnvironment.current().get_block_id(shape)
            if block_idx is None:
                # Check if this is a symbolic expression with block sizes
                shape_str = self._get_shape_string(shape)
                compacted_shapes.append(CompactedShape(shape_str, [idx], []))
            else:
                block_size = DeviceFunction.current().block_size_var(block_idx)
                if block_size is None:
                    block_size = "1"
                compacted_shapes.append(CompactedShape(block_size, [idx], [block_idx]))
        for strategy in self.strategies:
            compacted_shapes = strategy.compact_shape(compacted_shapes)
        return compacted_shapes

    def _get_shape_string(self, shape: SymIntLike) -> str:
        """Get string representation of a shape"""
        # Extract sympy expression
        if isinstance(shape, torch.SymInt):
            expr = shape._sympy_()
        elif isinstance(shape, sympy.Expr):
            expr = shape
        else:
            return self.strategies[0].fn.literal_expr(shape)

        # Try to map block symbols to their variable names
        mapped_expr = DeviceFunction.current().try_map_block_symbols_to_vars(expr)
        if mapped_expr is not None:
            # Use a dedicated tl.constexpr argument for any mapped shape expression.
            # This avoids emitting helper calls (e.g., triton_helpers.div_floor_integer)
            # in contexts that require compile-time constants such as tl.reshape shapes.
            df = DeviceFunction.current()
            const_name = df.new_var("_SHAPE_DIM")
            # Define on host using the original expression so origins are known.
            df.constexpr_arg_with_host_def(const_name, expr)
            return const_name

        # Fallback: use literal expression if mapping failed
        return self.strategies[0].fn.literal_expr(shape)

    def shape_str(self, shape: ShapeLike) -> str:
        compacted_shapes = self._compact_shape(shape)
        result = [s.size_str for s in compacted_shapes]
        return f"[{', '.join(result)}]"

    def expand_str(self, shape: ShapeLike, i: int) -> str:
        if len(shape) == 0 and i == 0:
            return ""
        assert 0 <= i < len(shape), f"Invalid index {i} for shape {shape}"
        compacted_shapes = self._compact_shape(shape)
        result = []
        for dim in compacted_shapes:
            if i in dim.user_indices:
                result.append(":")
            else:
                result.append("None")
        if result == [":"]:
            return ""
        return f"[{', '.join(result)}]"

    def expand_dims_str(self, shape: ShapeLike, start_idx: int, num_dims: int) -> str:
        """Generate expansion string for multi-dimensional tensor indexers.

        For a tensor with `num_dims` dimensions starting at `start_idx` in the output
        shape, generates an indexing string that preserves those dimensions and adds
        None for all other positions.

        For example, with shape=[1, 8, 16], start_idx=0, num_dims=2:
            Returns "[:, :, None]" - preserves positions 0,1 and adds None for position 2
        """
        if len(shape) == 0:
            return ""
        end_idx = start_idx + num_dims
        assert 0 <= start_idx < len(shape), (
            f"Invalid start_idx {start_idx} for shape {shape}"
        )
        assert end_idx <= len(shape), f"Invalid end_idx {end_idx} for shape {shape}"

        compacted_shapes = self._compact_shape(shape)
        result = []
        for dim in compacted_shapes:
            # Check if any of this dim's user_indices fall in our range [start_idx, end_idx)
            in_range = any(start_idx <= idx < end_idx for idx in dim.user_indices)
            if in_range:
                result.append(":")
            else:
                result.append("None")
        # If result is all colons, no expansion needed
        if all(r == ":" for r in result):
            return ""
        return f"[{', '.join(result)}]"

    def get_reduction_strategy(self, block_idx: int) -> ReductionStrategy:
        strategy = self.block_id_to_strategy[(block_idx,)]
        assert isinstance(strategy, ReductionStrategy)
        return strategy

    def user_size(self, block_index: int) -> sympy.Expr:
        """The user-visible size of the block index."""
        # This only does something special for reduction loops, only need to check for 1D loop
        strategy = self.block_id_to_strategy.get((block_index,))
        if strategy is None:
            return CompileEnvironment.current().block_sizes[block_index].symbol()
        return strategy.user_size(block_index)
