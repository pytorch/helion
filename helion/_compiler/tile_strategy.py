from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
import weakref

import sympy
import torch

from .. import exc
from .._compat import shape_env_size_hint
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .device_function import DeviceFunction
from .host_function import HostFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import XYZProgramIDs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .inductor_lowering import CodegenState

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


class ThreadAxisTracker:
    """Tracks thread axis assignments for block dimensions during codegen."""

    __slots__ = ("sizes", "block_axes")

    def __init__(self) -> None:
        self.sizes: dict[int, int] = {}
        self.block_axes: dict[int, int] = {}

    def record(self, block_idx: int, axis: int, size: int) -> None:
        """Record a thread axis mapping for a single block dimension."""
        self.sizes[axis] = max(self.sizes.get(axis, 1), size)
        self.block_axes[block_idx] = axis

    def record_all(self, block_ids: list[int], axis: int, size: int) -> None:
        """Record the same thread axis mapping for all block dimensions."""
        self.sizes[axis] = size
        for block_id in block_ids:
            self.block_axes[block_id] = axis


@dataclasses.dataclass
class LoopDimInfo:
    begin_var_name: str | None = None
    begin_expr: sympy.Expr | None = None
    end_var_name: str | None = None
    end_expr: sympy.Expr | None = None

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        shape_env = CompileEnvironment.current().shape_env
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return shape_env_size_hint(shape_env, expected) == shape_env_size_hint(
            shape_env, self.end_expr
        )


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]
    thread_axis_sizes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    block_thread_axes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EmitPipelineLoopState(DeviceLoopOrGridState):
    """State for emit_pipeline-based loops on TPU (Pallas backend)."""

    body_fn_name: str
    body_fn_def: ast.FunctionDef | None = None
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    pipeline_call: ast.AST | None = None
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ForiLoopState(DeviceLoopOrGridState):
    """State for fori_loop-based loops on TPU (Pallas backend).

    Uses jax.lax.fori_loop with pltpu.make_async_copy for manual DMA control.
    When ``use_dma=False``, skips DMA and accesses HBM refs directly via
    ``pl.ds`` slicing (used when inner block shapes violate TPU DMA alignment).
    """

    body_fn_name: str
    loop_var_name: str  # The fori_loop index variable (e.g., "_j")
    use_dma: bool = True
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_vmem: dict[str, str] = dataclasses.field(default_factory=dict)
    _tensor_to_sem: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def add_lane_loop(self, block_id: int, lane_var: str, extent: int) -> None:
        self.lane_loops.append((lane_var, extent))
        self.lane_loop_blocks.add(block_id)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [
                create(
                    ast.For,
                    target=create(ast.Name, id=lane_var, ctx=ast.Store()),
                    iter=expr_from_string(f"range({extent})"),
                    body=wrapped,
                    orelse=[],
                    type_comment=None,
                )
            ]
        return wrapped


@dataclasses.dataclass
class PersistentReductionState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [
                create(
                    ast.For,
                    target=create(ast.Name, id=lane_var, ctx=ast.Store()),
                    iter=expr_from_string(f"range({extent})"),
                    body=wrapped,
                    orelse=[],
                    type_comment=None,
                )
            ]
        return wrapped


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_ids: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"offset_{block_idx}", dce=True)
            for block_idx in block_ids
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def supports_index_rank_expansion(self) -> bool:
        """Whether index expressions produced by this strategy are tensor-shaped."""
        return True

    def thread_axes_used(self) -> int:
        return 0

    def thread_block_sizes(self) -> list[int]:
        """Return the thread block size for each thread axis this strategy uses."""
        return []

    def thread_block_size_exprs(self) -> list[str]:
        """Return per-axis thread block sizes as launch-time expressions."""
        return [str(size) for size in self.thread_block_sizes()]

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            config.range_warp_specializes, block_idx, None
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if "tensor_descriptor" in config.indexing:
            # Tensor descriptor + multi-stage pipelines in addition to unrolling tend to cause
            # CUDA "misaligned address" or "unspecified launch failure" errors.
            if range_num_stages > 0:
                range_num_stages = 0
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            # Unrolling can cause CUDA IMA with pipelining
            # We want to ensure new step size + pipeline is within bounds
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            step = range_unroll_factor * block_size
            last_offset = ((loop_numel - 1) // block_size) * block_size
            remainder = loop_numel - last_offset
            range_num_stages = min(
                max(1, int(math.ceil(remainder / step))), range_num_stages
            )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            kwargs.append(f"data_partition_factor={dpf_value}")

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()

        # Allow backend to override the range expression entirely
        backend_range = env.backend.range_str(begin, end, step)
        if backend_range is not None:
            return backend_range

        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self,
        state: CodegenState,
        use_proxy_ends: bool = False,
        ends_override: list[object] | None = None,
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds.

        Args:
            state: The codegen state
            use_proxy_ends: If True, use proxy_ends from state.proxy_args (for device loops)
            ends_override: If provided, use these ends instead of block_sizes.numel (for data-dependent bounds)
        """
        env = CompileEnvironment.current()
        block_id_to_info = {}

        def begin_to_ast(value: object) -> ast.AST:
            if isinstance(value, ast.AST):
                return value
            if isinstance(value, int):
                return expr_from_string(repr(value))
            if isinstance(value, sympy.Expr):
                return expr_from_string(DeviceFunction.current().sympy_expr(value))
            if isinstance(value, torch.SymInt):
                return begin_to_ast(value._sympy_())
            if isinstance(value, torch.Tensor):
                tensor_arg = DeviceFunction.current().tensor_arg(value)
                return expr_from_string(env.backend.scalar_load_expr(tensor_arg.name))
            raise NotImplementedError(f"{type(value)} is not implemented.")

        def normalize_dim_values(value: object) -> list[object]:
            if isinstance(value, (list, tuple, torch.Size)):
                return list(value)
            return [value]

        begin_values: list[object] | None = None
        proxy_begins: list[object] | None = None
        if isinstance(state.ast_args, (list, tuple)):
            if len(state.ast_args) >= 2 and isinstance(state.ast_args[1], list):
                begin_values = state.ast_args[1]
        if isinstance(state.proxy_args, (list, tuple)):
            if len(state.proxy_args) >= 2 and isinstance(
                state.proxy_args[1], (list, tuple, torch.Size)
            ):
                proxy_begins = normalize_dim_values(state.proxy_args[1])
                if begin_values is None:
                    begin_values = proxy_begins
            elif len(state.proxy_args) >= 2:
                begin_arg, end_arg = state.proxy_args[:2]
                if end_arg is None:
                    proxy_begins = [0] * len(normalize_dim_values(begin_arg))
                else:
                    proxy_begins = normalize_dim_values(begin_arg)
                if begin_values is None:
                    begin_values = proxy_begins

        if use_proxy_ends:
            _, _, proxy_ends, _, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for idx, (block_idx, end) in enumerate(
                zip(self.block_ids, proxy_ends, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=None,
                    end_expr=end_expr,
                )
        elif ends_override is not None:
            # Data-dependent bounds: use the provided ends
            for idx, (block_id, end) in enumerate(
                zip(self.block_ids, ends_override, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                    end_var_name = state.sympy_expr(end_expr)
                else:
                    # Tensor (data-dependent) - end_expr is None, but we still need end_var
                    end_expr = None
                    end_var_name = None
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )
        else:
            for idx, block_id in enumerate(self.block_ids):
                block_size_info = env.block_sizes[block_id]
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if block_size_info.size is None:
                    # Data-dependent bound - skip numel, it will be handled elsewhere
                    end_expr = None
                    end_var_name = None
                else:
                    end_expr = block_size_info.numel
                    end_var_name = state.sympy_expr(end_expr)
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )

        return block_id_to_info

    def _setup_block_size_constexpr(
        self, state: CodegenState, block_size_var: str, block_size: SymIntLike
    ) -> None:
        """Helper to setup constexpr block size variable on host."""
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def _get_data_dependent_numel(
        self, state: CodegenState, end: object, begin: object
    ) -> sympy.Expr | str:
        """Get numel for data-dependent bounds using the tensor end value.

        When the tile bound is a tensor (data-dependent), we need to pass
        the tensor to the kernel and use it to compute the number of elements.
        Returns either a sympy.Expr or a string expression.
        """
        from .device_function import DeviceFunction

        device_function = DeviceFunction.current()

        if isinstance(end, torch.Tensor):
            # For tensor bounds, we need to add it as a kernel argument
            # and load the scalar value
            tensor_arg = device_function.tensor_arg(end)
            end_expr = CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        elif isinstance(end, (int, torch.SymInt)):
            end_expr = device_function.sympy_expr(_to_sympy(end))
        else:
            raise NotImplementedError(f"Unsupported end type: {type(end)}")

        if begin == 0:
            # Simple case: numel = end
            return end_expr  # type: ignore[return-value]
        if isinstance(begin, torch.Tensor):
            begin_arg = device_function.tensor_arg(begin)
            begin_expr = CompileEnvironment.current().backend.scalar_load_expr(
                begin_arg.name
            )
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        if isinstance(begin, (int, torch.SymInt)):
            begin_expr = device_function.sympy_expr(_to_sympy(begin))
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        raise NotImplementedError(f"Unsupported begin type: {type(begin)}")

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := env.block_sizes[block_id].from_config(
                    state.config
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end

    def _compute_thread_axis_offset(
        self,
        active_device_loops: dict[int, list[DeviceLoopOrGridState]],
    ) -> int:
        """Compute the starting thread axis for the next strategy.

        Counts axes already claimed by active device loops, reserving at
        least one axis for reduction strategies when the backend places
        reductions first.
        """
        from .reduction_strategy import ReductionStrategy

        env = CompileEnvironment.current()
        seen: set[int] = set()
        active_reduction_axes = 0
        active_non_reduction_axes = 0
        for loops in active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                axes = loop_state.strategy.thread_axes_used()
                if env.backend.reduction_axis_first() and isinstance(
                    loop_state.strategy, ReductionStrategy
                ):
                    active_reduction_axes += axes
                else:
                    active_non_reduction_axes += axes

        if not env.backend.reduction_axis_first():
            return active_non_reduction_axes + active_reduction_axes

        has_reduction_strategy = any(
            isinstance(strategy, ReductionStrategy) and strategy.thread_axes_used() > 0
            for strategy in self.fn.tile_strategy.strategies
        )
        reserved_reduction_axes = max(
            1 if has_reduction_strategy else 0, active_reduction_axes
        )
        return reserved_reduction_axes + active_non_reduction_axes

    def select_pid_strategy(self) -> ProgramIDs:
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        if pid_type == "persistent_blocked":
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    # pyrefly: ignore [bad-override]
    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        if not env.backend.force_tile_mask() and env.known_multiple(
            functools.reduce(
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        ):
            self._mask_var = None
        else:
            self._mask_var: str | None = self.new_var("mask", dce=True)
        self._offsets_var = self.new_var("offsets", dce=True)

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def thread_axes_used(self) -> int:
        return int(self._uses_thread_axis())

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis() or not isinstance(self.block_size, int):
            return []
        return [self.block_size]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        if isinstance(self.block_size, int):
            return [str(self.block_size)]
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        return [bs_var]

    def _uses_thread_axis(self) -> bool:
        return not (isinstance(self.block_size, int) and self.block_size == 1)

    def _numel_str(self, state: CodegenState, value: sympy.Expr | str) -> str:
        if isinstance(value, str):
            return value
        return state.sympy_expr(value)

    def _range_trip_count(
        self,
        begin: object,
        end: object,
        step: object | None,
    ) -> sympy.Expr | str:
        return self._range_numel_expr(begin, end, step)

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        if isinstance(value, torch.Tensor):
            tensor_arg = DeviceFunction.current().tensor_arg(value)
            return CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        if isinstance(value, str):
            return value
        raise NotImplementedError(f"{type(value)} is not implemented.")

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _extract_root_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        assert len(state.proxy_args) == 3
        if state.proxy_args[1] is None:
            begins: list[object] = [0] * len(self.block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins_arg = state.proxy_args[0]
            begins = (
                list(begins_arg)
                if isinstance(begins_arg, (list, tuple))
                else [begins_arg]
            )
            ends_arg = state.proxy_args[1]
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _extract_device_loop_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        if len(state.ast_args) == 5:
            _, begins_arg, ends_arg, _, steps_arg = state.ast_args
        else:
            _, begins_arg, ends_arg, _ = state.ast_args
            steps_arg = None
        begins = (
            list(begins_arg) if isinstance(begins_arg, (list, tuple)) else [begins_arg]
        )
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(steps_arg, len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _codegen_common(
        self,
        state: CodegenState,
        *,
        begins: list[object] | None = None,
        ends: list[object] | None = None,
        steps: list[object | None] | None = None,
    ) -> tuple[str, str, sympy.Expr | str, list[ast.AST]]:
        offsets_var = self._offsets_var
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        if begins is None:
            begins = [0] * len(block_ids)
        if ends is None:
            ends = [env.block_sizes[block_id].numel for block_id in block_ids]
        if steps is None:
            steps = [None] * len(block_ids)
        total_numel: sympy.Expr | str = sympy.S.One
        statements = []

        # pyrefly: ignore [bad-assignment]
        for i, (block_idx, begin, end, step) in enumerate(
            self._reorder([*zip(block_ids, begins, ends, steps, strict=True)])
        ):
            cute_scalar_tile = (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
                and step not in (None, 1)
            )
            numel = (
                self._range_numel_expr(begin, end, None)
                if cute_scalar_tile
                else self._range_trip_count(begin, end, step)
            )
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({self._numel_str(state, total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({self._numel_str(state, numel)})"
            step_expr = self._expr_str(step) if step not in (None, 1) else None
            if step_expr is not None and not (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
            ):
                expr = f"({expr}) * ({step_expr})"
            if begin != 0:
                expr = f"({self._expr_str(begin)}) + ({expr})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            if isinstance(total_numel, str) or isinstance(numel, str):
                total_numel = (
                    f"({self._numel_str(state, total_numel)})"
                    f" * ({self._numel_str(state, numel)})"
                )
            else:
                assert isinstance(total_numel, sympy.Expr)
                assert isinstance(numel, sympy.Expr)
                total_numel = sympy.Mul(total_numel, numel)

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            mask_terms = [f"{offsets_var} < ({self._numel_str(state, total_numel)})"]
            thread_mask = env.backend.thread_in_tile_mask_expr(
                block_size_var, axis=self._flat_thread_axis()
            )
            if thread_mask is not None:
                mask_terms.insert(0, f"({thread_mask})")
            mask_expr = " and ".join(mask_terms)
            statements.append(statement_from_string(f"{mask_var} = {mask_expr}"))
        # pyrefly: ignore [bad-return]
        return block_size_var, offsets_var, total_numel, statements

    def _flat_thread_axis(self) -> int:
        """Compute the thread axis for this flattened strategy.

        For CuTe, reduction strategies occupy earlier axes.
        """
        return self._compute_thread_axis_offset(self.fn.codegen.active_device_loops)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        assert state.ast_args is None

        from .ast_extension import ExtendedAST
        from .type_propagation import GridIndexType
        from .type_propagation import IterType
        from .type_propagation import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        scalar_grid_loop = False
        if isinstance(type_info, IterType):
            inner = (
                type_info.inner.unpack()
                if isinstance(type_info.inner, SequenceType)
                else [type_info.inner]
            )
            scalar_grid_loop = len(inner) == 1 and isinstance(inner[0], GridIndexType)

        if (
            scalar_grid_loop
            and len(self.block_ids) == 1
            and len(state.proxy_args) == 3
            and not isinstance(state.proxy_args[0], (list, tuple))
            and (
                state.proxy_args[1] is None
                or not isinstance(state.proxy_args[1], (list, tuple))
            )
            and not isinstance(state.proxy_args[2], (list, tuple))
        ):

            def _range_bound_to_sympy(value: object) -> sympy.Expr:
                assert isinstance(value, (int, torch.SymInt, sympy.Expr))
                return _to_sympy(value)

            step = state.proxy_args[2]
            if step not in (None, 1):
                block_id = self.block_ids[0]
                if state.proxy_args[1] is None:
                    begin = 0
                    end = state.proxy_args[0]
                else:
                    begin = state.proxy_args[0]
                    end = state.proxy_args[1]
                    if isinstance(begin, (list, tuple)):
                        assert len(begin) == 1
                        begin = begin[0]
                    if isinstance(end, (list, tuple)):
                        assert len(end) == 1
                        end = end[0]
                begin_expr = _range_bound_to_sympy(begin)
                end_expr = _range_bound_to_sympy(end)
                step_expr = _range_bound_to_sympy(step)
                trip_count = (
                    f"(({state.sympy_expr(end_expr)}) - ({state.sympy_expr(begin_expr)}) + "
                    f"({state.sympy_expr(step_expr)}) - 1) // ({state.sympy_expr(step_expr)})"
                )

                env = CompileEnvironment.current()
                dtype = env.index_type()
                pid_var = state.device_function.new_var("pid_flat", dce=True)
                offsets_var = self._offsets_var
                block_size_var = self.block_size_var(-1)
                self._setup_block_size_constexpr(state, block_size_var, self.block_size)
                pids = self.select_pid_strategy()
                if isinstance(state.device_function.pid, ForEachProgramID):
                    pids.shared_pid_var = state.device_function.pid.shared_pid_var
                pids.append(PIDInfo(pid_var, block_size_var, trip_count, block_id))
                state.add_statement(
                    env.backend.arange_expr(
                        offsets_var,
                        pid_var,
                        block_size_var,
                        dtype,
                        axis=self._flat_thread_axis(),
                    )
                )
                index_var = self.index_var(block_id)
                state.add_statement(
                    f"{index_var} = ({state.sympy_expr(begin_expr)}) + ({offsets_var}) * ({state.sympy_expr(step_expr)})"
                )
                mask_var = self.mask_var(-1)
                if mask_var is not None:
                    mask_terms = [f"{offsets_var} < ({trip_count})"]
                    thread_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=self._flat_thread_axis()
                    )
                    if thread_mask is not None:
                        mask_terms.insert(0, f"({thread_mask})")
                    state.add_statement(
                        statement_from_string(
                            f"{mask_var} = {' and '.join(mask_terms)}"
                        )
                    )
                pids.codegen(state)
                if isinstance(state.device_function.pid, ForEachProgramID):
                    shared_pid = state.device_function.pid
                    shared_pid.cases.append(pids)
                    shared_pid.codegen(state)
                else:
                    state.device_function.set_pid(pids)
                tracker = ThreadAxisTracker()
                if self._uses_thread_axis() and isinstance(self.block_size, int):
                    tracker.record_all(
                        self.block_ids, self._flat_thread_axis(), self.block_size
                    )
                return DeviceGridState(
                    self,
                    block_id_to_info=self._create_block_id_info_dict(
                        state, ends_override=[end]
                    ),
                    thread_axis_sizes=tracker.sizes,
                    block_thread_axes=tracker.block_axes,
                )
        begins, ends, steps = self._extract_root_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))

        state.add_statement(
            env.backend.arange_expr(
                offsets_var,
                pid_var,
                block_size_var,
                dtype,
                axis=self._flat_thread_axis(),
            )
        )
        state.codegen.statements_stack[-1].extend(statements)

        pids.codegen(state)

        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        begins, ends, steps = self._extract_device_loop_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()
        lid = self.new_var("lid")
        numel_str = self._numel_str(state, total_numel)
        end_var = env.backend.cdiv_expr(numel_str, block_size_var, is_device=True)
        arange_expr = env.backend.arange_expr(
            offsets_var, lid, block_size_var, dtype, axis=self._flat_thread_axis()
        )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=(
                body := [
                    statement_from_string(arange_expr),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            if not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            ):
                flatten_loops.disable_block_id(block_ids[0])
                continue
            for i, j in itertools.pairwise(block_ids):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    flatten_loops.disable_block_id(block_ids[0])
                    break

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # Keep axis structure intact for multi-phase kernels (e.g., barrier) to
        # avoid mismatched ranks in downstream reductions.
        if len(HostFunction.current().device_ir.root_ids) > 1:
            return shapes

        env = CompileEnvironment.current()
        # Filter out unit-sized blocks that don't need compacting
        compact_block_ids = [
            block_id
            for block_id in self.block_ids
            if not (
                isinstance(env.block_sizes[block_id].size, int)
                and env.block_sizes[block_id].size == 1
            )
        ]
        if not compact_block_ids:
            return shapes

        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            # Check if this starts our flattened sequence
            if len(shape.block_ids) != 1 or shape.block_ids[0] != compact_block_ids[0]:
                output.append(shape)
                continue

            # Try to collect the full sequence
            group_shapes = [shape]
            found_complete_sequence = True
            for expected in compact_block_ids[1:]:
                if (
                    shape_queue
                    and len(shape_queue[0].block_ids) == 1
                    and shape_queue[0].block_ids[0] == expected
                ):
                    group_shapes.append(shape_queue.popleft())
                else:
                    # Partial match - don't combine
                    found_complete_sequence = False
                    output.extend(group_shapes)
                    break

            if found_complete_sequence:
                # Full match - combine into one
                for s in group_shapes[1:]:
                    shape = shape.combine(s)
                output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    # pyrefly: ignore [bad-override]
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def _uses_thread_axis(self, block_size: SymIntLike) -> bool:
        return not (isinstance(block_size, int) and block_size == 1)

    def thread_axes_used(self) -> int:
        return sum(
            1 for block_size in self.block_size if self._uses_thread_axis(block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if self._uses_thread_axis(bs) and isinstance(bs, int):
                sizes.append(bs)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis(bs):
                continue
            if isinstance(bs, int):
                exprs.append(str(bs))
            else:
                bs_var = self.block_size_var(block_id)
                if bs_var is None:
                    return []
                exprs.append(bs_var)
        return exprs

    def _thread_axis_offset(self, state: CodegenState) -> int:
        return self._compute_thread_axis_offset(state.codegen.active_device_loops)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis(block_size_by_id[block_id]):
                axis += 1
        return mapping

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _root_grid_steps(self, state: CodegenState) -> list[object | None]:
        from .ast_extension import ExtendedAST
        from .type_propagation import GridIndexType
        from .type_propagation import IterType
        from .type_propagation import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        assert isinstance(type_info, IterType)
        inner = (
            type_info.inner.unpack()
            if isinstance(type_info.inner, SequenceType)
            else [type_info.inner]
        )
        if not all(isinstance(value, GridIndexType) for value in inner):
            return [None] * len(self.block_ids)
        return self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        return ast.unparse(self._to_ast(value))

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        elif (
            isinstance(pids, FlatProgramIDs)
            and env.backend.name == "pallas"
            and len(block_ids) >= 2
        ):
            pids = XYZProgramIDs()

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                step_var = state.codegen.lift(step_ast, dce=True, prefix="step").id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
            axis = thread_axis_offset + thread_axis_map[block_idx]
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis(block_size)
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.grid_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            state.add_statement(f"{index_var} = {idx_expr}")
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        # Only use ends_override if there are data-dependent (tensor) bounds
        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                cast_expr = CompileEnvironment.current().backend.ast_to_dtype_expr(
                    "{value}", to_dtype
                )
                return expr_from_string(cast_expr, value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        if isinstance(x, torch.Tensor):
            # Handle tensor values (for data-dependent bounds)
            # For scalar tensors, we need to load the value using tl.load
            from .device_function import DeviceFunction

            tensor_arg = DeviceFunction.current().tensor_arg(x)
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_load_expr(tensor_arg.name)
            )
        if isinstance(x, str):
            # Already a string expression (for data-dependent numel)
            return expr_from_string(x)
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis(block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.loop_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            extra_body = [
                statement_from_string(f"{index_var} = {idx_expr}"),
            ]
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            # pyrefly: ignore [unsupported-operation]
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
    ) -> ast.stmt | None:
        env = CompileEnvironment.current()
        if (
            not env.backend.force_tile_mask()
            and env.block_sizes[block_idx].known_multiple(block_size)
            and not env.is_jagged_tile(block_idx)
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )

        if env.is_jagged_tile(block_idx):
            jagged_tile_parents_ast = state.ast_args[3]
            jagged_tile_parents_proxy = state.proxy_args[3]
            assert isinstance(jagged_tile_parents_ast, list)
            assert isinstance(jagged_tile_parents_proxy, list)
            # We guarantee the first lifted loop input is the jagged_tile parent tensor.
            jagged_tile_parent = jagged_tile_parents_ast[0]
            jagged_tile_block_size = env.block_sizes[block_idx].var
            jagged_tile_parent_proxy = jagged_tile_parents_proxy[0]
            assert isinstance(jagged_tile_parent_proxy, torch.Tensor)
            jagged_tile_parent_block_size = jagged_tile_parent_proxy.size(0)
            assert isinstance(jagged_tile_parent_block_size, torch.SymInt)
            env.jagged_tile_mask_shapes[block_idx] = [
                jagged_tile_parent_block_size,
                jagged_tile_block_size,
            ]
            return statement_from_string(
                f"{mask_var} = ({index_var})[None,:] < {{parent}}[:,None]",
                parent=self._to_ast(jagged_tile_parent),
            )

        return statement_from_string(
            f"{mask_var} = ({index_var}) < {{end}}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class CuteNDTileStrategy(NDTileStrategy):
    """CuTe N-D tile strategy using the standard tile pipeline."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
        num_threads: list[int] | None = None,
        mma_mode: bool = False,
        inactive_block_ids: set[int] | None = None,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order, l2_grouping)
        assert isinstance(block_size, list)
        if num_threads is None:
            num_threads = [0 for _ in block_ids]
        assert len(num_threads) == len(block_ids)
        self.num_threads = num_threads
        self.mma_mode = mma_mode
        self.inactive_block_ids = inactive_block_ids or set()
        self._lane_var_by_block: dict[int, str] = {}
        if not mma_mode:
            for block_id, nt, bs in zip(
                block_ids, num_threads, block_size, strict=True
            ):
                if block_id in self.inactive_block_ids:
                    continue
                static_bs = self._configured_block_size_int(bs)
                if (
                    nt > 0
                    and static_bs is not None
                    and static_bs > nt
                    and static_bs % nt == 0
                ):
                    self._lane_var_by_block[block_id] = self.fn.new_var(
                        f"lane_{block_id}"
                    )

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        if isinstance(block_size, int):
            return block_size
        env = CompileEnvironment.current()
        resolved_block_id = env.resolve_block_id(block_size)
        if resolved_block_id is not None:
            configured_size = env.block_sizes[resolved_block_id].from_config(
                self.fn.config
            )
            if isinstance(configured_size, int):
                return configured_size
        block_size_expr = _to_sympy(block_size)
        block_size_expr = env.specialize_expr(block_size_expr)
        if getattr(block_size_expr, "free_symbols", None):
            return None
        return int(block_size_expr)

    def _elements_per_thread_for_block(self, block_id: int) -> int:
        """Elements per thread for *block_id* (derived from num_threads)."""
        if block_id in self.inactive_block_ids:
            return 1
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return 1
        bs = self._configured_block_size_int(self.block_size[idx])
        assert isinstance(bs, int)  # validated by _thread_extent_for_axis
        return bs // nt

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        if block_id in self.inactive_block_ids:
            return 1
        if self.mma_mode:
            return 1  # MMA handles element distribution, no CUDA threads needed
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return block_size
        resolved_block_size = block_size
        if not isinstance(resolved_block_size, int):
            static_block_size = self._configured_block_size_int(resolved_block_size)
            if static_block_size is None:
                raise exc.BackendUnsupported(
                    "cute",
                    "num_threads requires static ND block sizes for cute",
                )
            resolved_block_size = static_block_size
        if resolved_block_size % nt != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute axis "
                    f"{block_id}: {resolved_block_size} is not divisible by {nt}"
                ),
            )
        return nt

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        if block_id in self.inactive_block_ids:
            return False
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                axis += 1
        return mapping

    def thread_axes_used(self) -> int:
        return sum(
            1
            for block_idx, block_size in zip(
                self.block_ids, self.block_size, strict=True
            )
            if self._uses_thread_axis_for_block(block_idx, block_size)
        )

    def _static_thread_extent_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> int | None:
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        if isinstance(thread_extent, int):
            return thread_extent
        return self._configured_block_size_int(thread_extent)

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            thread_extent = self._thread_extent_for_axis(
                block_id, block_size_by_id[block_id]
            )
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                static_extent = thread_extent
                if not isinstance(static_extent, int):
                    static_extent = self._configured_block_size_int(static_extent)
                if isinstance(static_extent, int):
                    sizes.append(static_extent)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, bs):
                continue
            thread_extent = self._thread_extent_for_axis(block_id, bs)
            if isinstance(thread_extent, int):
                exprs.append(str(thread_extent))
                continue
            if not isinstance(bs, torch.SymInt):
                return []
            bs_var = self.block_size_var(block_id)
            if bs_var is None:
                return []
            elements_per_thread = self._elements_per_thread_for_block(block_id)
            if elements_per_thread == 1:
                exprs.append(bs_var)
            else:
                exprs.append(f"({bs_var}) // {elements_per_thread}")
        return exprs

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if not self._lane_var_by_block:
            return super().codegen_grid(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        lane_setup_statements: list[ast.AST] = []
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                step_var = state.codegen.lift(step_ast, dce=True, prefix="step").id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")

            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {elements_per_thread}"
                static_thread_extent = self._static_thread_extent_for_block(
                    block_idx, block_size
                )
                if isinstance(static_thread_extent, int):
                    tracker.record(block_idx, axis, static_thread_extent)
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                idx_expr = f"{idx_expr} + cutlass.Int32({lane_var})"
            lane_setup_statements.append(
                statement_from_string(f"{index_var} = {idx_expr}")
            )

            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                lane_setup_statements.append(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = [
            (
                self._lane_var_by_block[block_id],
                self._elements_per_thread_for_block(block_id),
            )
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_setup_statements=lane_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if not self._lane_var_by_block and not self.mma_mode:
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = user_body = []
        lane_loops = [
            (
                self._lane_var_by_block[block_id],
                self._elements_per_thread_for_block(block_id),
            )
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        for lane_var, extent in reversed(lane_loops):
            lane_for = create(
                ast.For,
                target=create(ast.Name, id=lane_var, ctx=ast.Store()),
                iter=expr_from_string(f"range({extent})"),
                body=body,
                orelse=[],
                type_comment=None,
            )
            body = [lane_for]
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        index_setup: list[ast.stmt] = []
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {elements_per_thread}"
                static_thread_extent = self._static_thread_extent_for_block(
                    block_idx, block_size
                )
                if isinstance(static_thread_extent, int):
                    tracker.record(block_idx, axis, static_thread_extent)
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                idx_expr = f"{idx_expr} + cutlass.Int32({lane_var})"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def supports_index_rank_expansion(self) -> bool:
        return False


class CuteFlattenedTileStrategy(FlattenedTileStrategy):
    """Flattened CuTe strategy: scalar index per thread over a flattened tile."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        num_threads: int = 0,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self._num_threads = num_threads
        self._lane_var: str | None = None
        if num_threads > 0 and isinstance(block_size, int) and num_threads < block_size:
            self._lane_var = self.new_var("lane", dce=False)

    @property
    def _elements_per_thread(self) -> int:
        """Elements per thread (derived from num_threads and block_size)."""
        if self._num_threads == 0:
            return 1
        assert isinstance(self.block_size, int)
        return self.block_size // self._num_threads

    def _thread_extent(self) -> SymIntLike:
        if self._num_threads == 0:
            return self.block_size
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                "cute",
                "num_threads requires static flattened block sizes for cute",
            )
        if self.block_size % self._num_threads != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute: "
                    f"{self.block_size} is not divisible by {self._num_threads}"
                ),
            )
        return self._num_threads

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if not isinstance(thread_extent, int):
            return []
        return [thread_extent]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if isinstance(thread_extent, int):
            return [str(thread_extent)]
        if not isinstance(self.block_size, torch.SymInt):
            return []
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        if self._num_threads == 0:
            return [bs_var]
        return [f"({bs_var}) // {self._elements_per_thread}"]

    def _uses_thread_axis(self) -> bool:
        thread_extent = self._thread_extent()
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if self._lane_var is None:
            return super().codegen_grid(state)

        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + cutlass.Int32({self._lane_var})"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        axis = self._flat_thread_axis()
        state.add_statement(
            f"{offsets_base_var} = ({pid_var}) * ({block_size_var}) + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {self._elements_per_thread}"
        )
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)
        block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = []
        if self._lane_var is not None:
            lane_loops = [(self._lane_var, self._elements_per_thread)]
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self.block_ids) if lane_loops else set(),
            lane_setup_statements=lane_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if self._lane_var is None:
            return super().codegen_device_loop(state)

        env = CompileEnvironment.current()
        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + cutlass.Int32({self._lane_var})"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        lid = self.new_var("lid")
        end_var = env.backend.cdiv_expr(
            state.sympy_expr(total_numel), block_size_var, is_device=True
        )
        axis = self._flat_thread_axis()
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        user_body[:0] = lane_setup_statements
        if self._lane_var is not None:
            lane_for = create(
                ast.For,
                target=create(ast.Name, id=self._lane_var, ctx=ast.Store()),
                iter=expr_from_string(f"range({self._elements_per_thread})"),
                body=body,
                orelse=[],
                type_comment=None,
            )
            body = [lane_for]
        body[:0] = [
            statement_from_string(
                f"{offsets_base_var} = {lid} * ({block_size_var}) + cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {self._elements_per_thread}"
            )
        ]
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def offset_var(self, block_idx: int) -> str:
        return self._offsets_var

    def supports_index_rank_expansion(self) -> bool:
        return False


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
