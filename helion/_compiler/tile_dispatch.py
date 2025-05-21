from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._compiler.reduction_strategy import ReductionStrategy
from helion._compiler.tile_strategy import CompactedShape
from helion._compiler.tile_strategy import DeviceGridState
from helion._compiler.tile_strategy import DeviceLoopState
from helion._compiler.tile_strategy import FlattenedTileStrategy
from helion._compiler.tile_strategy import NDGridTileStrategy
from helion._compiler.tile_strategy import NDTileStrategy
from helion._compiler.tile_strategy import TileStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy
    import torch

    from helion import Config
    from helion._compiler.inductor_lowering import CodegenState

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
        self.block_indices_to_strategy: dict[tuple[int, ...], TileStrategy] = {}
        self._add_loop_strategies(fn, config)
        self._add_reduction_strategies(fn, config)

    def _add_loop_strategies(self, fn: DeviceFunction, config: Config) -> None:
        device_ir = HostFunction.current().device_ir
        for block_indices in device_ir.grid_block_indices:
            self._add_loop_strategy(block_indices, fn, config)
        for graph in device_ir.graphs:
            if isinstance(graph, ForLoopGraphInfo) and not isinstance(
                graph, ReductionLoopGraphInfo
            ):
                block_indices = [*graph.block_indices]
                self._add_loop_strategy(block_indices, fn, config)

    def _add_loop_strategy(
        self, block_indices: list[int], fn: DeviceFunction, config: Config
    ) -> None:
        env = CompileEnvironment.current()
        block_size_infos = [env.block_sizes[i] for i in block_indices]
        loop_order = block_size_infos[0].get_order(config, len(block_size_infos))
        if block_size_infos[0].is_grid():
            strategy: TileStrategy = NDGridTileStrategy(
                fn,
                block_indices,
                loop_order=loop_order,
            )
        elif block_size_infos[0].is_flattened(config):
            strategy: TileStrategy = FlattenedTileStrategy(
                fn,
                block_indices,
                block_size=block_size_infos[0].from_config_assert(config),
                loop_order=loop_order,
            )
        else:
            strategy = NDTileStrategy(
                fn,
                block_indices,
                block_size=[bs.from_config_assert(config) for bs in block_size_infos],
                loop_order=loop_order,
                l2_grouping=block_size_infos[0].l2_grouping(config),
            )
        self.strategies.append(strategy)
        self.block_indices_to_strategy[tuple(block_indices)] = strategy

    def _add_reduction_strategies(self, fn: DeviceFunction, config: Config) -> None:
        env = CompileEnvironment.current()
        rdims = [bs.block_size_idx for bs in env.block_sizes if bs.reduction]
        reduction_loops = collections.deque(config.reduction_loops)
        for rdim_index, rdim_spec in zip(
            rdims, env.config_spec.reduction_loop_specs, strict=True
        ):
            reduction_loop = reduction_loops.popleft() if rdim_spec.allow_loop else None
            if reduction_loop is None:
                strategy: TileStrategy = PersistentReductionStrategy(fn, rdim_index)
            else:
                strategy = LoopedReductionStrategy(fn, rdim_index, reduction_loop)
            self.strategies.append(strategy)
            self.block_indices_to_strategy[(rdim_index,)] = strategy
        assert not reduction_loops

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        strategy = self.block_indices_to_strategy[tuple(block_indices)]
        strategy.codegen_grid(state)
        for other_strategy in self.strategies:
            if other_strategy is not strategy:
                other_strategy.codegen_preamble(state)
        state.codegen.set_active_loops(DeviceGridState(strategy))

    def codegen_device_loop(
        self, state: CodegenState, block_indices: list[int]
    ) -> DeviceLoopState:
        strategy = self.block_indices_to_strategy[tuple(block_indices)]
        return strategy.codegen_device_loop(state)

    def _compact_shape(self, shapes: ShapeLike) -> list[CompactedShape]:
        compacted_shapes = []
        for idx, shape in enumerate(shapes):
            block_idx = TileStrategy.get_block_index(shape)
            if block_idx is None:
                compacted_shapes.append(
                    CompactedShape(self.strategies[0].fn.literal_expr(shape), [idx], [])
                )
            else:
                block_size = DeviceFunction.current().block_size_var(block_idx)
                if block_size is None:
                    block_size = "1"
                compacted_shapes.append(CompactedShape(block_size, [idx], [block_idx]))
        for strategy in self.strategies:
            compacted_shapes = strategy.compact_shape(compacted_shapes)
        return compacted_shapes

    def shape_str(self, shape: ShapeLike) -> str:
        compacted_shapes = self._compact_shape(shape)
        result = [s.size_str for s in compacted_shapes]
        return f"[{', '.join(result)}]"

    def expand_str(self, shape: ShapeLike, i: int) -> str:
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

    def get_reduction_strategy(self, block_idx: int) -> ReductionStrategy:
        strategy = self.block_indices_to_strategy[(block_idx,)]
        assert isinstance(strategy, ReductionStrategy)
        return strategy

    def user_size(self, block_index: int) -> sympy.Expr:
        """The user-visible size of the block index."""
        # This only does something special for reduction loops, only need to check for 1D loop
        strategy = self.block_indices_to_strategy.get((block_index,))
        if strategy is None:
            return CompileEnvironment.current().block_sizes[block_index].symbol()
        return strategy.user_size(block_index)
