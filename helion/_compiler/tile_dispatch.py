from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._compiler.reduction_strategy import ReductionStrategy
from helion._compiler.tile_strategy import CompactedShape
from helion._compiler.tile_strategy import FlattenedTileStrategy
from helion._compiler.tile_strategy import NDTileStrategy
from helion._compiler.tile_strategy import TileStrategy

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    import sympy
    import torch

    from helion import Config
    from helion._compiler.device_function import DeviceFunction
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
        env = CompileEnvironment.current()
        specs = env.config_spec.block_size_specs
        block_size_idx = iter(
            [bs.block_size_idx for bs in env.block_sizes if not bs.reduction]
        )
        block_sizes = config.block_sizes
        loop_orders = collections.deque(config.loop_orders)
        assert len(block_sizes) == len(specs)
        self.block_index_to_strategy: dict[int, TileStrategy] = {}
        self.strategies: list[TileStrategy] = []
        for spec, block_size in zip(specs, block_sizes):
            block_indices = [next(block_size_idx) for _ in range(len(spec))]
            if spec.allow_reorder:
                loop_order = loop_orders.popleft()
            else:
                loop_order = [*range(len(spec))]
            strategy_cls = (
                FlattenedTileStrategy if isinstance(block_size, int) else NDTileStrategy
            )
            strategy = strategy_cls(fn, block_indices, spec, block_size, loop_order)
            self.strategies.append(strategy)
            for idx in block_indices:
                self.block_index_to_strategy[idx] = strategy
        assert not loop_orders
        rdims = [bs.block_size_idx for bs in env.block_sizes if bs.reduction]
        for rdim_index in rdims:
            # TODO(jansel): add looped reduction config choices
            strategy = PersistentReductionStrategy(fn, rdim_index)
            self.strategies.append(strategy)
            self.block_index_to_strategy[rdim_index] = strategy

    def offset_var(self, block_idx: int) -> str:
        return self.block_index_to_strategy[block_idx].offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.block_index_to_strategy[block_idx].index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        return self.block_index_to_strategy[block_idx].mask_var(block_idx)

    def need_mask(self, block_idx: int) -> bool:
        return self.block_index_to_strategy[block_idx].mask_var(block_idx) is not None

    def block_size_var(self, block_idx: int) -> str | None:
        return self.block_index_to_strategy[block_idx].block_size_var(block_idx)

    def codegen_grid(self, state: CodegenState, block_indices: list[int]) -> None:
        strategy = self.block_index_to_strategy[block_indices[0]]
        assert strategy.block_indices == block_indices
        strategy.codegen_grid(state)
        for other_strategy in self.strategies:
            if other_strategy is not strategy:
                other_strategy.codegen_preamble(state)

    def codegen_device_loop(
        self, state: CodegenState, block_indices: list[int]
    ) -> tuple[ast.For, list[ast.AST]]:
        strategy = self.block_index_to_strategy[block_indices[0]]
        assert strategy.block_indices == block_indices
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
                block_size = self.block_size_var(block_idx)
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
        assert 0 <= i < len(shape)
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
        strategy = self.block_index_to_strategy[block_idx]
        assert isinstance(strategy, ReductionStrategy)
        return strategy

    def user_size(self, block_index: int) -> sympy.Expr:
        """The user-visible size of the block index."""
        strategy = self.block_index_to_strategy[block_index]
        return strategy.user_size(block_index)
