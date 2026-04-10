from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from .._compat import shape_env_size_hint
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
    from collections.abc import ItemsView
    from collections.abc import Sequence

    from .. import Config
    from .inductor_lowering import CodegenState

    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


class BlockIDStrategyMapping:
    def __init__(self) -> None:
        self._block_id_to_strategy: dict[tuple[int, ...], TileStrategy] = {}
        self._block_id_to_any_strategy: dict[int, TileStrategy] = {}

    def __setitem__(self, block_ids: tuple[int, ...], strategy: TileStrategy) -> None:
        self._block_id_to_strategy[block_ids] = strategy
        for block_id in block_ids:
            self._block_id_to_any_strategy.setdefault(block_id, strategy)

    def __getitem__(self, block_ids: tuple[int, ...]) -> TileStrategy:
        return self._block_id_to_strategy[block_ids]

    def get(
        self, block_ids: tuple[int, ...], default: TileStrategy | None = None
    ) -> TileStrategy | None:
        return self._block_id_to_strategy.get(block_ids, default)

    def get_any(self, block_id: int) -> TileStrategy | None:
        strategy = self._block_id_to_strategy.get((block_id,))
        if strategy is None:
            strategy = self._block_id_to_any_strategy.get(block_id)
        return strategy

    def items(self) -> ItemsView[tuple[int, ...], TileStrategy]:
        return self._block_id_to_strategy.items()

    def clear(self) -> None:
        self._block_id_to_strategy.clear()
        self._block_id_to_any_strategy.clear()


class TileStrategyDispatch:
    def __init__(
        self,
        fn: DeviceFunction,
        config: Config,
    ) -> None:
        super().__init__()
        self.strategies: list[TileStrategy] = []
        self.block_id_to_strategy = BlockIDStrategyMapping()
        self._add_loop_strategies(fn, config)
        self._add_reduction_strategies(fn, config)

    def _add_loop_strategies(self, fn: DeviceFunction, config: Config) -> None:
        device_ir = HostFunction.current().device_ir
        for block_ids in device_ir.grid_block_ids:
            self._add_loop_strategy(block_ids, fn, config)
        for graph in fn.codegen.codegen_graphs:
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
        self._register_strategy(block_ids, strategy)

    def _register_strategy(self, block_ids: list[int], strategy: TileStrategy) -> None:
        self.strategies.append(strategy)
        self.block_id_to_strategy[tuple(block_ids)] = strategy

    def _add_reduction_strategies(self, fn: DeviceFunction, config: Config) -> None:
        env = CompileEnvironment.current()
        max_threads = env.backend.max_reduction_threads()
        rdims = [bs.block_id for bs in env.block_sizes if bs.reduction]
        reduction_loop_block_ids = set(
            env.config_spec.reduction_loops.valid_block_ids()
        )
        for block_id in rdims:
            reduction_loop = env.config_spec.reduction_loops.config_get(
                config.reduction_loops, block_id, None
            )
            # Only rolled reduction dimensions can use LoopedReductionStrategy.
            # Non-rolled dimensions must stay persistent so graph selection and
            # strategy selection remain consistent.
            if max_threads is not None and block_id in reduction_loop_block_ids:
                numel = env.block_sizes[block_id].numel
                if isinstance(numel, sympy.Integer):
                    size_hint = int(numel)
                elif isinstance(numel, sympy.Expr):
                    size_hint = shape_env_size_hint(env.shape_env, numel)
                else:
                    size_hint = env.size_hint(numel)
                if reduction_loop is None:
                    if size_hint > max_threads:
                        # Too many elements for a single warp; force looped
                        reduction_loop = max_threads
                elif reduction_loop > max_threads:
                    reduction_loop = max_threads
            if reduction_loop is None:
                strategy: TileStrategy = PersistentReductionStrategy(fn, block_id)
            else:
                strategy = LoopedReductionStrategy(fn, block_id, reduction_loop)
            self._register_strategy([block_id], strategy)

    def codegen_grid(self, state: CodegenState, block_ids: list[int]) -> None:
        strategy = self.block_id_to_strategy[tuple(block_ids)]
        state.codegen.active_device_loops.clear()
        grid_state = strategy.codegen_grid(state)
        state.codegen.set_active_loops(grid_state)
        for other_strategy in self.strategies:
            if other_strategy is not strategy:
                other_strategy.codegen_preamble(state)

    def codegen_device_loop(
        self, state: CodegenState, block_ids: list[int]
    ) -> DeviceLoopState:
        strategy = self.block_id_to_strategy[tuple(block_ids)]
        return strategy.codegen_device_loop(state)

    def _compact_shape(self, shapes: ShapeLike) -> list[CompactedShape]:
        compacted_shapes = []
        for idx, shape in enumerate(shapes):
            block_idx = CompileEnvironment.current().resolve_block_id(shape)
            if block_idx is None:
                # Check if this is a symbolic expression with block sizes
                shape_str = self._get_shape_string(shape)
                compacted_shapes.append(CompactedShape(shape_str, [idx], []))
            else:
                strategy = self.block_id_to_strategy.get((block_idx,))
                if strategy is None:
                    strategy = self.block_id_to_strategy.get_any(block_idx)
                if strategy is not None:
                    block_size = strategy.block_size_var(block_idx)
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
        return f"[{', '.join(self.shape_dims(shape))}]"

    def shape_dims(self, shape: ShapeLike) -> list[str]:
        compacted_shapes = self._compact_shape(shape)
        return [s.size_str for s in compacted_shapes]

    def supports_index_rank_expansion(self) -> bool:
        return all(
            strategy.supports_index_rank_expansion() for strategy in self.strategies
        )

    def _ordered_strategies_for_branch(
        self, branch: list[TileStrategy]
    ) -> list[TileStrategy]:
        """Order strategies for thread axis assignment.

        For CuTe, reduction strategies must come first (axis 0) so that
        reduction threads are within the same warp for warp-level reductions.
        """
        env = CompileEnvironment.current()
        if not env.backend.reduction_axis_first():
            return branch
        reductions = [s for s in branch if isinstance(s, ReductionStrategy)]
        non_reductions = [s for s in branch if not isinstance(s, ReductionStrategy)]
        return reductions + non_reductions

    def _strategy_branches(self) -> list[list[TileStrategy]]:
        """Return strategy branches that execute mutually exclusively."""
        device_ir = HostFunction.current().device_ir
        num_grids = len(device_ir.grid_block_ids)
        grid_strategies = self.strategies[:num_grids]

        if num_grids <= 1:
            return [self.strategies]

        loop_strategies = self.strategies[num_grids:]
        branches: list[list[TileStrategy]] = []
        for i, grid_strat in enumerate(grid_strategies):
            branch: list[TileStrategy] = [grid_strat]
            grid_max = max(grid_strat.block_ids)
            next_min = (
                min(grid_strategies[i + 1].block_ids)
                if i + 1 < num_grids
                else float("inf")
            )
            for ls in loop_strategies:
                if all(grid_max < bid < next_min for bid in ls.block_ids):
                    branch.append(ls)
            branches.append(branch)
        return branches

    def thread_axis_for_strategy(self, target: TileStrategy) -> int | None:
        """Return the starting thread-axis index for a strategy in its branch."""
        for branch in self._strategy_branches():
            if target not in branch:
                continue
            axis = 0
            for strategy in self._ordered_strategies_for_branch(branch):
                if strategy is target:
                    return axis
                axis += strategy.thread_axes_used()
        return None

    def thread_axis_for_block_id(self, target_block_id: int) -> int | None:
        """Return the launch thread axis assigned to a specific logical block id."""
        for strategy in self.strategies:
            if target_block_id not in strategy.block_ids:
                continue
            base_axis = self.thread_axis_for_strategy(strategy)
            if base_axis is None:
                return None
            local_axis_map = getattr(strategy, "_thread_axis_map", None)
            if callable(local_axis_map):
                axis_map = local_axis_map()
                if isinstance(axis_map, dict):
                    local_axis = axis_map.get(target_block_id)
                    if local_axis is not None:
                        return base_axis + local_axis
            if strategy.thread_axes_used() > 0:
                return base_axis
        return None

    def _iter_strategy_thread_axes(
        self, strategy: TileStrategy
    ) -> list[tuple[int, int, int | None, str | None]]:
        ordered_block_ids = getattr(strategy, "loop_order", None)
        if ordered_block_ids is None:
            block_id_order = list(strategy.block_ids)
        else:
            block_id_order = [strategy.block_ids[i] for i in ordered_block_ids]
        axis_map_fn = getattr(strategy, "_thread_axis_map", None)
        axis_map = axis_map_fn() if callable(axis_map_fn) else None
        total_axes = strategy.thread_axes_used()
        size_iter = iter(strategy.thread_block_sizes())
        expr_iter = iter(strategy.thread_block_size_exprs())
        result: list[tuple[int, int, int | None, str | None]] = []
        for idx, block_id in enumerate(block_id_order):
            local_axis = (
                axis_map.get(block_id, idx) if isinstance(axis_map, dict) else idx
            )
            if idx + 1 < len(block_id_order):
                next_axis = (
                    axis_map.get(block_id_order[idx + 1], local_axis + 1)
                    if isinstance(axis_map, dict)
                    else idx + 1
                )
            else:
                next_axis = total_axes
            if local_axis >= next_axis:
                continue
            size = next(size_iter, None)
            expr = next(expr_iter, None)
            result.append((block_id, local_axis, size, expr))
        return result

    def thread_extent_for_block_id(self, target_block_id: int) -> int | None:
        """Return the live thread extent for a specific logical block id."""
        for strategy in self.strategies:
            if target_block_id not in strategy.block_ids:
                continue
            if isinstance(strategy, ReductionStrategy):
                count = strategy._reduction_thread_count()
                return count if count > 0 else None
            for block_id, _local_axis, size, _expr in self._iter_strategy_thread_axes(
                strategy
            ):
                if block_id == target_block_id:
                    return size
        return None

    def thread_axis_sizes(self) -> dict[int, int]:
        dims = self.thread_block_dims()
        return {axis: size for axis, size in enumerate(dims) if size > 1}

    def _thread_extent_expr_for_block_id(self, target_block_id: int) -> str | None:
        """Return the launch-time thread extent expression for a specific block id."""
        for strategy in self.strategies:
            if target_block_id not in strategy.block_ids:
                continue
            for block_id, _local_axis, _size, expr in self._iter_strategy_thread_axes(
                strategy
            ):
                if block_id == target_block_id:
                    return expr
        return None

    def thread_block_dims(self) -> tuple[int, int, int]:
        """Compute the CUDA thread block dims from all strategies.

        When there are multiple grid entries (ForEach pattern), each branch
        is mutually exclusive and shares the same thread axes.  We compute
        per-branch dims and take the elementwise max.
        """
        branches = self._strategy_branches()
        dims = [1, 1, 1]
        for branch in branches:
            branch_dims = [1, 1, 1]
            ordered = self._ordered_strategies_for_branch(branch)
            for strategy in ordered:
                for block_id in strategy.block_ids:
                    axis = self.thread_axis_for_block_id(block_id)
                    size = self.thread_extent_for_block_id(block_id)
                    if axis is None or size is None or size <= 1 or axis >= len(dims):
                        continue
                    branch_dims[axis] = max(branch_dims[axis], size)
            for axis, size in enumerate(branch_dims):
                dims[axis] = max(dims[axis], size)
        return dims[0], dims[1], dims[2]

    def thread_block_dim_exprs(self) -> tuple[str, str, str] | None:
        """Compute launch block dims as expressions for single-branch kernels.

        For kernels with dynamic block-size constexprs, this provides symbolic
        launch dims (e.g. ``_BLOCK_SIZE_0``) when static tracking cannot infer
        thread extents.
        """
        branches = self._strategy_branches()
        if len(branches) != 1:
            return None
        dims = ["1", "1", "1"]
        for strategy in self._ordered_strategies_for_branch(branches[0]):
            for block_id in strategy.block_ids:
                axis = self.thread_axis_for_block_id(block_id)
                size_expr = self._thread_extent_expr_for_block_id(block_id)
                if axis is None or size_expr is None:
                    continue
                if axis >= len(dims):
                    return None
                current = dims[axis]
                if current == "1":
                    dims[axis] = size_expr
                elif current != size_expr:
                    if current.isdigit() and size_expr.isdigit():
                        dims[axis] = str(max(int(current), int(size_expr)))
                    else:
                        return None
        return dims[0], dims[1], dims[2]

    def expand_str(self, shape: ShapeLike, i: int) -> str:
        if not self.supports_index_rank_expansion():
            return ""
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

    def jagged_tile_expand_str(self, src_shape: ShapeLike, dst_shape: ShapeLike) -> str:
        """Return suffix to transform src to dst using permute + None indexing.

        Examples:
            (u0, u2) -> (u0, u2, u1) => "[:, :, None]"
            (u2, u0) -> (u0, u2, u1) => ".permute(1, 0)[:, :, None]"
        """
        if not self.supports_index_rank_expansion():
            return ""
        if len(src_shape) == 0:
            return ""
        assert len(src_shape) <= len(dst_shape), (src_shape, dst_shape)

        env = CompileEnvironment.current()

        # Map each source dim to a unique destination dim with equal symbolic size.
        src_to_dst: list[int] = []
        used_dst: set[int] = set()
        for src_dim in src_shape:
            match: int | None = None
            for dst_i, dst_dim in enumerate(dst_shape):
                if dst_i in used_dst:
                    continue
                if env.known_equal(src_dim, dst_dim):
                    match = dst_i
                    break
            assert match is not None, (
                f"Cannot map src dim {src_dim} into dst shape {dst_shape} "
                f"from src shape {src_shape}"
            )
            src_to_dst.append(match)
            used_dst.add(match)

        # Reorder source axes so they match destination axis order.
        perm = sorted(range(len(src_shape)), key=lambda src_i: src_to_dst[src_i])

        parts: list[str] = []
        if perm != list(range(len(src_shape))):
            parts.append(f".permute({', '.join(str(i) for i in perm)})")

        # Add singleton dimensions where destination has extra axes.
        keep = set(src_to_dst)
        index_parts = [":" if i in keep else "None" for i in range(len(dst_shape))]
        if not all(x == ":" for x in index_parts):
            parts.append(f"[{', '.join(index_parts)}]")

        return "".join(parts)

    def expand_dims_str(self, shape: ShapeLike, start_idx: int, num_dims: int) -> str:
        """Generate expansion string for multi-dimensional tensor indexers.

        For a tensor with `num_dims` dimensions starting at `start_idx` in the output
        shape, generates an indexing string that preserves those dimensions and adds
        None for all other positions.

        For example, with shape=[1, 8, 16], start_idx=0, num_dims=2:
            Returns "[:, :, None]" - preserves positions 0,1 and adds None for position 2
        """
        if not self.supports_index_rank_expansion():
            return ""
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
