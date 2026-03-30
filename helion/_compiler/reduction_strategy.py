from __future__ import annotations

import ast
import logging
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._prims_common import get_computation_dtype

from .._compat import shape_env_size_hint
from ..autotuner.config_fragment import integer_power_of_two
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.layout import LayoutTag as _CuteLayoutTag
from .cute.layout_propagation import META_KEY as _CUTE_LAYOUT_META_KEY
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import install_inductor_kernel_handlers
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .tile_strategy import LoopDimInfo
from .tile_strategy import PersistentReductionState
from .tile_strategy import ThreadAxisTracker
from .tile_strategy import TileStrategy
from .tile_strategy import _to_sympy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

log = logging.getLogger(__name__)


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _cute_shared_memory_budget_bytes() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    default_shared = int(props.shared_memory_per_block)
    optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(default_shared, optin_shared)


def _log_cute_reduction_layout(state: CodegenState) -> None:
    """Log the CuTe layout annotation for the current reduction node, if any."""
    if state.fx_node is None:
        return
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.layout is None:
        return
    layout = constraint.layout
    log.debug(
        "cute reduction %s: layout tag=%s thread=%s value=%s",
        state.fx_node.name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _reduction_threads_from_annotation(state: CodegenState) -> int | None:
    """Read reduction thread count from the layout annotation, if available.

    Returns the thread count from the layout annotation when the node has
    a REDUCTION-tagged layout with a concrete integer thread count.
    Falls back to ``None`` so the caller can use ``reduction_threads_hint()``.
    """
    if state.fx_node is None:
        return None
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.layout is None:
        return None
    layout = constraint.layout
    if layout.tag != _CuteLayoutTag.REDUCTION:
        return None
    nt = layout.num_threads()
    if isinstance(nt, int) and nt > 0:
        return nt
    return None


def _cute_reduction_smem_bytes(num_elements: int, dtype: torch.dtype) -> int:
    return num_elements * torch.empty((), dtype=dtype).element_size()


class ReductionStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        mask_var: str | None,
        block_size_var: str | None,
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=[block_index],
        )
        self._mask_var = mask_var
        if block_size_var is not None:
            fn.block_size_var_cache[(block_index,)] = block_size_var

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    @property
    def block_index(self) -> int:
        return self.block_ids[0]

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].numel

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        return shapes

    def _reduction_thread_count(self) -> int:
        """Return threads used for this reduction on thread-aware backends."""
        return 0

    def thread_axes_used(self) -> int:
        return 1 if self._reduction_thread_count() > 0 else 0

    def thread_block_sizes(self) -> list[int]:
        count = self._reduction_thread_count()
        return [count] if count > 0 else []

    def _reduction_block_has_lane_loops(self) -> bool:
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and current_grid.has_lane_loops()
            and self.block_index in current_grid.lane_loop_blocks
        ):
            return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceGridState)
                    and loop_state.has_lane_loops()
                    and self.block_index in loop_state.lane_loop_blocks
                ):
                    return True
        return False

    def _reduction_block_is_serial(self) -> bool:
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if (
                isinstance(loop_state, DeviceLoopState)
                and self.block_index not in loop_state.block_thread_axes
            ):
                return True
        return False

    def _reduction_block_has_live_thread_axis(self) -> bool:
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and self.block_index in current_grid.block_thread_axes
        ):
            return True
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if self.block_index in loop_state.block_thread_axes:
                return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if self.block_index in loop_state.block_thread_axes:
                    return True
        return False

    def _planned_thread_dims(self) -> tuple[int, int, int]:
        return self.fn.tile_strategy.thread_block_dims()

    def _block_has_live_thread_axis(
        self, block_id: int, extent: int | None = None
    ) -> bool:
        axis = self.fn.tile_strategy.thread_axis_for_block_id(block_id)
        if axis is None:
            return False
        planned_dims = self._planned_thread_dims()
        if axis >= len(planned_dims):
            return False
        live_extent = planned_dims[axis]
        return live_extent > 1 and not (extent is not None and extent > live_extent)

    def _reduction_dim_has_live_thread_axis(
        self,
        fake_input: torch.Tensor,
        dim: int,
    ) -> bool:
        env = CompileEnvironment.current()
        normalized_dim = dim if dim >= 0 else fake_input.ndim + dim
        if not (0 <= normalized_dim < fake_input.ndim):
            return False
        block_id = env.resolve_block_id(fake_input.size(normalized_dim))
        if block_id is None:
            return False
        extent = self.fn.tile_strategy.thread_extent_for_block_id(block_id)
        return self._block_has_live_thread_axis(block_id, extent)

    def _get_thread_axis(self) -> int:
        """Compute the thread axis index for this reduction strategy.

        Some backends place reduction strategies first so reduction threads share
        a warp. Others keep the natural strategy order.
        """
        env = CompileEnvironment.current()
        if (axis := self.fn.tile_strategy.thread_axis_for_strategy(self)) is not None:
            return axis
        if env.backend.reduction_axis_first():
            axis = 0
            for strategy in self.fn.tile_strategy.strategies:
                if strategy is self:
                    break
                if isinstance(strategy, ReductionStrategy):
                    axis += strategy.thread_axes_used()
            return axis
        axis = 0
        for strategy in self.fn.tile_strategy.strategies:
            if strategy is self:
                break
            axis += strategy.thread_axes_used()
        return axis

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        raise NotImplementedError

    def call_reduction_function(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        backend = CompileEnvironment.current().backend
        if backend.is_indexed_reduction(reduction_type):
            index_var = self.index_var(self.block_index)
            return self.call_indexed_reduction(
                input_name,
                self.broadcast_str(index_var, fake_input, dim),
                reduction_type,
                dim,
                fake_output,
            )
        return backend.reduction_expr(
            input_name,
            reduction_type,
            dim,
            block_size_var=self.block_size_var(self.block_index),
        )

    def _index_init_expr(self, block_size_var: str, dtype: str, block_idx: int) -> str:
        env = CompileEnvironment.current()
        backend = env.backend
        size = env.block_sizes[block_idx].size
        if isinstance(size, int) and size == 0:
            return backend.reduction_index_zero_expr(dtype)
        if isinstance(size, torch.SymInt) and env.known_equal(size, 0):
            return backend.reduction_index_zero_expr(dtype)
        return backend.reduction_index_expr(
            block_size_var, dtype, block_idx, axis=self._get_thread_axis()
        )

    def call_indexed_reduction(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
    ) -> str:
        env = CompileEnvironment.current()
        return env.backend.argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            dim,
            fake_output.dtype,
            block_size_var=self.block_size_var(self.block_index),
            index_dtype=env.index_dtype,
        )

    def maybe_reshape(
        self,
        expr: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        size = [*fake_input.size()]
        size.pop(dim)
        if [*fake_output.size()] == size:
            return expr
        shape = self.fn.tile_strategy.shape_str([*fake_output.size()])
        return CompileEnvironment.current().backend.reshape_expr(expr, shape)

    def broadcast_str(self, base: str, fake_input: torch.Tensor, dim: int) -> str:
        input_size = [*fake_input.size()]
        expand = self.fn.tile_strategy.expand_str(input_size, dim)
        shape = self.fn.tile_strategy.shape_str(input_size)
        return CompileEnvironment.current().backend.broadcast_to_expr(
            f"{base}{expand}", shape
        )


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        from .device_ir import ReductionLoopGraphInfo

        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        if isinstance(numel, (int, sympy.Integer)) and integer_power_of_two(int(numel)):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_RDIM_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
        else:
            self._thread_count = 0
        self._synthetic_cute_lane_var: str | None = None
        self._synthetic_cute_lane_extent = 1
        is_graph_reduction_dim = any(
            isinstance(graph, ReductionLoopGraphInfo) and block_index in graph.block_ids
            for graph in fn.codegen.codegen_graphs
        )
        if (
            env.backend.name == "cute"
            and not is_graph_reduction_dim
            and self._thread_count > 0
        ):
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            padded_size = next_power_of_2(max(1, size_hint))
            if padded_size > self._thread_count:
                self._synthetic_cute_lane_var = fn.new_var(
                    f"synthetic_lane_{block_index}",
                    dce=False,
                )
                self._synthetic_cute_lane_extent = padded_size // self._thread_count

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        backend = env.backend
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        mask_var = self._mask_var
        block_size_var = self.block_size_var(self.block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            if isinstance(numel, sympy.Integer):
                # Static size - issue statement immediately
                stmt = statement_from_string(
                    f"{block_size_var} = {next_power_of_2(int(numel))}"
                )
                state.codegen.host_statements.append(stmt)
            else:
                # Check for block size dependencies
                block_mapping, _ = find_block_size_symbols(numel)
                if block_mapping:
                    # Defer issuing statement until block sizes are known
                    state.device_function.deferred_rdim_defs.append(
                        (block_size_var, numel)
                    )
                else:
                    # No dependencies - issue statement immediately
                    expr_str = HostFunction.current().sympy_expr(numel)
                    stmt = statement_from_string(
                        f"{block_size_var} = {backend.next_power_of_2_host_expr(expr_str)}"
                    )
                    state.codegen.host_statements.append(stmt)
        current_grid = state.codegen.current_grid_state
        synthetic_lane_var = self._synthetic_cute_lane_var
        if synthetic_lane_var is not None and current_grid is not None:
            axis = self._get_thread_axis()
            current_grid.add_lane_loop(
                block_idx,
                synthetic_lane_var,
                self._synthetic_cute_lane_extent,
            )
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1),
                self._thread_count,
            )
            current_grid.block_thread_axes[block_idx] = axis
            index_expr = (
                f"({self._index_init_expr(block_size_var, env.index_type(), block_idx)})"
                f" + cutlass.Int32({synthetic_lane_var}) * {self._thread_count}"
            )
            current_grid.lane_setup_statements.append(
                statement_from_string(f"{index_var} = {index_expr}")
            )
            if mask_var is not None:
                current_grid.lane_setup_statements.append(
                    statement_from_string(
                        f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                    )
                )
        else:
            state.add_statement(
                f"{index_var} = {self._index_init_expr(block_size_var, env.index_type(), block_idx)}"
            )
            if mask_var is not None:
                state.add_statement(
                    f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                )
        # Extract end_var_name from the numel expression
        from .tile_strategy import LoopDimInfo

        end_var_name = self.fn.sympy_expr(numel)
        block_id_to_info = {
            self.block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(
                self.block_index, self._get_thread_axis(), self._thread_count
            )
        state.codegen.push_active_loops(
            PersistentReductionState(
                self,
                block_id_to_info=block_id_to_info,
                thread_axis_sizes=tracker.sizes,
                block_thread_axes=tracker.block_axes,
            )
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend
        numel = env.block_sizes[self.block_index].numel
        if isinstance(numel, sympy.Integer) and numel == 0:
            default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
            assert isinstance(default, (float, int, bool))
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                backend.full_expr(shape_dims, constant_repr(default), fake_output.dtype)
            )
        expr = self.call_reduction_function(
            input_name,
            reduction_type,
            dim,
            fake_input,
            fake_output,
        )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))


class LoopedReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int,
    ) -> None:
        env = CompileEnvironment.current()
        if env.known_multiple(env.block_sizes[block_index].numel, block_size):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_REDUCTION_BLOCK_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"roffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"rindex_{block_index}", dce=True)
        self.block_size = block_size
        assert block_size > 1
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            self._thread_count = next_power_of_2(min(block_size, max_threads))
        else:
            self._thread_count = 0

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        block_index = self.block_index
        numel = env.block_sizes[block_index].numel
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self.block_size!r}")
            )
        body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)}"
            ),
        ]
        if (mask_var := self._mask_var) is not None:
            body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {state.sympy_expr(numel)}"
                )
            )

        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(
                    state.config,
                    [self.block_index],
                    begin="0",
                    end=state.sympy_expr(numel),
                    step=block_size_var,
                ),
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        # Extract end_var_name from the actual numel expression used in the range()
        from .tile_strategy import LoopDimInfo

        end_var_name = state.sympy_expr(numel)
        block_id_to_info = {
            block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(block_index, self._get_thread_axis(), self._thread_count)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        with install_inductor_kernel_handlers(state.codegen, {}):
            env = CompileEnvironment.current()
            backend = env.backend
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_input.size()])
            acc_dtype = get_computation_dtype(fake_input.dtype)  # promote fp16 to fp32
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            assert state.fx_node is not None
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            acc_full = backend.full_expr(shape_dims, constant_repr(default), acc_dtype)
            device_loop.outer_prefix.append(
                statement_from_string(f"{acc} = {acc_full}")
            )
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if not backend.is_indexed_reduction(reduction_type):
                combine_expr = backend.reduction_combine_expr(
                    reduction_type, acc, input_name, acc_dtype
                )
                state.add_statement(f"{acc} = {combine_expr}")
                expr = self.call_reduction_function(
                    acc, reduction_type, dim, fake_input, fake_output
                )
            else:
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = env.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = {backend.reduction_index_init_expr(shape_dims, index_dtype)}"
                    )
                )
                index = self.broadcast_str(
                    self.index_var(self.block_index), fake_input, dim
                )
                for stmt in backend.argreduce_loop_update_statements(
                    reduction_type=reduction_type,
                    acc=acc,
                    acc_index=acc_index,
                    value=input_name,
                    index=index,
                ):
                    state.add_statement(stmt)
                expr = self.call_indexed_reduction(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                )
            # Ensure the final reduction result matches torch.* dtype semantics
            expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
            expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
            device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))

            # Optional: emit a dtype static assert right after the assignment when enabled
            if env.settings.debug_dtype_asserts:
                device_loop.outer_suffix.append(
                    statement_from_string(
                        f"tl.static_assert({result}.dtype == {_dtype_str(fake_output.dtype)})"
                    )
                )
            return expr_from_string(result)


class BlockReductionStrategy(ReductionStrategy):
    """This is used when we are reducing over a tile rather than an entire tensor."""

    def __init__(
        self,
        state: CodegenState,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=state.device_function,
            block_index=block_index,
            mask_var=state.codegen.mask_var(block_index),
            block_size_var=None,
        )
        self.offset_vars[block_index] = "0"
        # Store reference to codegen to access existing index variables
        self._codegen = state.codegen

    def index_var(self, block_idx: int) -> str:
        # Use the existing index variable from the active device loop
        # instead of the newly created one from TileStrategy.__init__
        return self._codegen.index_var(block_idx)

    def _active_thread_layout(self) -> tuple[dict[int, int], dict[int, int]]:
        axis_sizes: dict[int, int] = {}
        block_axes: dict[int, int] = {}
        seen: set[int] = set()
        for loops in self._codegen.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                    continue
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
                block_axes.update(loop_state.block_thread_axes)
        current_grid = getattr(self._codegen, "current_grid_state", None)
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
            block_axes.update(current_grid.block_thread_axes)
        return block_axes, axis_sizes

    def _aliased_active_thread_axis(self, block_axes: dict[int, int]) -> int | None:
        env = CompileEnvironment.current()
        target_block = self.block_index
        for candidate_block_id, axis in block_axes.items():
            if candidate_block_id == target_block:
                return axis
            source = env.block_sizes[candidate_block_id].block_size_source
            value = getattr(source, "value", None)
            if isinstance(value, torch.SymInt):
                if env.get_block_id(value) == target_block:
                    return axis
            elif isinstance(value, int):
                target_size = env.block_sizes[target_block].size
                if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                    target_size, value
                ):
                    return axis
        return None

    def _aliased_strategy_block_id(self) -> int | None:
        env = CompileEnvironment.current()
        target_block = self.block_index
        for strategy in self.fn.tile_strategy.strategies:
            for candidate_block_id in strategy.block_ids:
                if candidate_block_id == target_block:
                    return candidate_block_id
                source = env.block_sizes[candidate_block_id].block_size_source
                value = getattr(source, "value", None)
                if isinstance(value, torch.SymInt):
                    if env.get_block_id(value) == target_block:
                        return candidate_block_id
                elif isinstance(value, int):
                    target_size = env.block_sizes[target_block].size
                    if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                        target_size, value
                    ):
                        return candidate_block_id
        return None

    def _strided_thread_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        default_value: float | bool,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        current_grid = getattr(self._codegen, "current_grid_state", None)
        allow_lane_axis_fallback = (
            isinstance(current_grid, DeviceGridState) and current_grid.has_lane_loops()
        )
        normalized_dim = dim if dim >= 0 else fake_input.ndim + dim

        def debug(*parts: object) -> None:
            return None

        def block_thread_extent_hint(block_id: int) -> int | None:
            extent = self.fn.tile_strategy.thread_extent_for_block_id(block_id)
            if extent is not None:
                return extent
            configured_threads = env.config_spec.num_threads.config_get(
                self.fn.config.num_threads, block_id, 0
            )
            if configured_threads > 0:
                return configured_threads
            configured_block_size = env.block_sizes[block_id].from_config(
                self.fn.config
            )
            return (
                configured_block_size
                if isinstance(configured_block_size, int)
                else None
            )

        def active_loop_states() -> list[DeviceLoopState | DeviceGridState]:
            loop_states: list[DeviceLoopState | DeviceGridState] = []
            seen: set[int] = set()
            for loops in self._codegen.active_device_loops.values():
                for loop_state in loops:
                    if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                        continue
                    key = id(loop_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    loop_states.append(loop_state)
            return loop_states

        loop_states = active_loop_states()
        info_by_block: dict[int, LoopDimInfo] = {}
        if isinstance(current_grid, DeviceGridState):
            info_by_block.update(current_grid.block_id_to_info)
        for loop_state in loop_states:
            for block_id, info in loop_state.block_id_to_info.items():
                info_by_block.setdefault(block_id, info)
        planned_dims = self._planned_thread_dims()
        active_thread_blocks: list[tuple[int, int, int, LoopDimInfo]] = []
        seen_thread_blocks: set[int] = set()
        active_block_axes, active_axis_sizes = self._active_thread_layout()
        active_block_ids = set(info_by_block) | set(active_block_axes)
        for block_id in active_block_ids:
            if block_id in seen_thread_blocks:
                continue
            axis = active_block_axes.get(block_id)
            if axis is None:
                continue
            live_extent = active_axis_sizes.get(axis, 1)
            if live_extent <= 1:
                continue
            extent = block_thread_extent_hint(block_id)
            if extent is None:
                extent = live_extent
            else:
                extent = min(extent, live_extent)
            if extent <= 1:
                continue
            if extent > live_extent:
                continue
            info = info_by_block.get(block_id)
            if info is None:
                size = env.block_sizes[block_id].size
                if not isinstance(size, (int, torch.SymInt)):
                    size = extent
                end_expr = _to_sympy(size)
                info = LoopDimInfo(
                    end_var_name=state.sympy_expr(end_expr),
                    end_expr=end_expr,
                )
            active_thread_blocks.append((block_id, axis, extent, info))
            seen_thread_blocks.add(block_id)
        active_thread_blocks.sort(key=operator.itemgetter(1, 0))
        active_block_axes = {
            block_id: axis for block_id, axis, _, _ in active_thread_blocks
        }
        active_axis_sizes: dict[int, int] = {}
        for _, axis, extent, _ in active_thread_blocks:
            active_axis_sizes[axis] = max(active_axis_sizes.get(axis, 1), extent)

        def resolve_tensor_dim_mapping() -> dict[int, tuple[int, int, int]]:
            mapping: dict[int, tuple[int, int, int]] = {}
            used_block_ids: set[int] = set()
            used_axes: set[int] = set()
            for dim_idx in range(fake_input.ndim):
                dim_size = fake_input.size(dim_idx)
                candidates: dict[tuple[int, int, int], int] = {}
                block_id = env.resolve_block_id(dim_size)
                if block_id is not None and block_id in active_block_axes:
                    axis = active_block_axes[block_id]
                    extent = block_thread_extent_hint(block_id)
                    if extent is not None:
                        candidates[(block_id, axis, extent)] = 0
                for candidate_block_id, axis, extent, info in active_thread_blocks:
                    matches_end = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and info.is_end_matching(dim_size)
                    matches_thread_extent = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and env.known_equal(dim_size, extent)
                    candidate_source = getattr(
                        env.block_sizes[candidate_block_id].block_size_source,
                        "value",
                        None,
                    )
                    matches_source_value = (
                        isinstance(dim_size, torch.SymInt)
                        and isinstance(candidate_source, torch.SymInt)
                        and candidate_source._sympy_() == dim_size._sympy_()
                    )
                    if (
                        not matches_end
                        and not matches_thread_extent
                        and not matches_source_value
                    ):
                        continue
                    priority = 3
                    if matches_source_value:
                        priority = 1
                    elif matches_end:
                        priority = 2
                    candidate = (
                        candidate_block_id,
                        axis,
                        extent,
                    )
                    previous = candidates.get(candidate)
                    if previous is None or priority < previous:
                        candidates[candidate] = priority
                chosen: tuple[int, int, int] | None = None
                ordered_candidates = sorted(
                    candidates.items(),
                    key=lambda item: (item[1], item[0][1], item[0][0]),
                )
                for candidate, _priority in ordered_candidates:
                    block_id, axis, _ = candidate
                    if block_id in used_block_ids or axis in used_axes:
                        continue
                    chosen = candidate
                    break
                if (
                    chosen is None
                    and allow_lane_axis_fallback
                    and dim_idx != normalized_dim
                ):
                    for candidate_block_id, axis, extent, _info in sorted(
                        active_thread_blocks, key=operator.itemgetter(1, 0)
                    ):
                        if candidate_block_id in used_block_ids or axis in used_axes:
                            continue
                        chosen = (candidate_block_id, axis, extent)
                        break
                if chosen is None and ordered_candidates:
                    chosen = ordered_candidates[0][0]
                if chosen is None:
                    continue
                mapping[dim_idx] = chosen
                used_block_ids.add(chosen[0])
                used_axes.add(chosen[1])
            return mapping

        if backend.name != "cute":
            debug("skip backend", backend.name)
            return None
        if backend.is_indexed_reduction(reduction_type):
            debug("skip indexed", reduction_type)
            return None
        if self._reduction_block_is_serial():
            debug("skip serial", self.block_index)
            return None
        if state.fx_node is not None:
            for arg in state.fx_node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                target_name = getattr(arg.target, "__name__", "")
                if any(
                    name in target_name
                    for name in ("sum", "prod", "mean", "amax", "amin")
                ):
                    debug("skip nested reduction arg", target_name)
                    return None
        if self._reduction_block_has_lane_loops():
            # Lane loops serialize part of the logical tile in Python rather
            # than mapping it to actual threads. Thread-reduction fast paths
            # assume every participating axis is backed by a live thread, so
            # they are invalid under active lane loops.
            debug("skip lane loops")
            return None

        tensor_dim_mapping = resolve_tensor_dim_mapping()
        mapped_block_ids = {block_id for block_id, _, _ in tensor_dim_mapping.values()}
        logical_axes = {
            axis for _, axis, _ in tensor_dim_mapping.values() if axis is not None
        }
        reduce_axis: int | None = None
        reduce_thread_extent: int | None = None
        if 0 <= normalized_dim < fake_input.ndim:
            mapping = tensor_dim_mapping.get(normalized_dim)
            if mapping is not None:
                _, reduce_axis, reduce_thread_extent = mapping

        block_axes = dict(active_block_axes)
        axis_sizes = dict(active_axis_sizes)
        if reduce_axis is not None and reduce_thread_extent is not None:
            axis_sizes[reduce_axis] = max(
                axis_sizes.get(reduce_axis, 1), reduce_thread_extent
            )
            if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[reduce_axis] = max(
                    self._codegen.max_thread_block_dims[reduce_axis],
                    reduce_thread_extent,
                )
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            aliased_block_id = self._aliased_strategy_block_id()
            if aliased_block_id is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_block_id(
                    aliased_block_id
                )
                reduce_thread_extent = block_thread_extent_hint(aliased_block_id)
                if reduce_axis is not None and reduce_thread_extent is not None:
                    if (
                        reduce_axis >= len(planned_dims)
                        or planned_dims[reduce_axis] <= 1
                        or reduce_thread_extent > planned_dims[reduce_axis]
                    ):
                        reduce_axis = None
                        reduce_thread_extent = None
                    else:
                        axis_sizes[reduce_axis] = max(
                            axis_sizes.get(reduce_axis, 1), reduce_thread_extent
                        )
                        if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                            self._codegen.max_thread_block_dims[reduce_axis] = max(
                                self._codegen.max_thread_block_dims[reduce_axis],
                                reduce_thread_extent,
                            )
        if reduce_axis is None:
            strategy = self.fn.tile_strategy.block_id_to_strategy.get(
                (self.block_index,)
            )
            if strategy is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduce_axis is not None:
                hint = _reduction_threads_from_annotation(state)
                if hint is None:
                    hint = backend.reduction_threads_hint(
                        self.block_size_var(self.block_index)
                    )
                if (
                    hint is not None
                    and reduce_axis < len(planned_dims)
                    and planned_dims[reduce_axis] > 1
                    and hint <= planned_dims[reduce_axis]
                ):
                    axis_sizes[reduce_axis] = max(axis_sizes.get(reduce_axis, 1), hint)
                else:
                    reduce_axis = None
        if reduce_axis is None:
            debug("skip no reduce axis", tuple(fake_input.size()), dim)
            return None
        logical_axes.add(reduce_axis)
        logical_axis_sizes: dict[int, int] = {}
        for block_id, axis, extent, _info in active_thread_blocks:
            if block_id in mapped_block_ids or block_id >= self.block_index:
                logical_axis_sizes[axis] = max(
                    logical_axis_sizes.get(axis, 1),
                    extent,
                )
        if reduce_axis not in logical_axis_sizes and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            reduce_size = axis_sizes.get(reduce_axis, 1)
            if reduce_thread_extent is None:
                reduce_size = max(
                    reduce_size,
                    self._codegen.max_thread_block_dims[reduce_axis],
                )
            logical_axis_sizes[reduce_axis] = reduce_size
        if not logical_axis_sizes:
            debug("skip no logical axis sizes", tuple(fake_input.size()), dim)
            return None
        for axis, size in logical_axis_sizes.items():
            if 0 <= axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[axis] = max(
                    self._codegen.max_thread_block_dims[axis], size
                )
        if reduce_thread_extent is None and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            logical_axis_sizes[reduce_axis] = max(
                logical_axis_sizes.get(reduce_axis, 1),
                self._codegen.max_thread_block_dims[reduce_axis],
            )

        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        reduce_extent = logical_axis_sizes.get(reduce_axis, 1)
        group_span = pre * reduce_extent
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            debug("skip no lane expr", tuple(fake_input.size()), dim)
            return None

        dtype = _dtype_str(fake_input.dtype)
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype)
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        tensor_thread_axes: set[int] = set()
        tensor_thread_footprint = 1
        for _block_id, axis, extent in tensor_dim_mapping.values():
            if axis is None or extent is None or axis in tensor_thread_axes:
                continue
            tensor_thread_axes.add(axis)
            tensor_thread_footprint *= extent
        if (
            reduce_axis is not None
            and reduce_thread_extent is not None
            and reduce_axis not in tensor_thread_axes
        ):
            tensor_thread_axes.add(reduce_axis)
            tensor_thread_footprint *= reduce_thread_extent
        actual_threads = 1
        planned_dims = self.fn.tile_strategy.thread_block_dims()
        for axis, (recorded, planned) in enumerate(
            zip(self._codegen.max_thread_block_dims, planned_dims, strict=True)
        ):
            if axis not in logical_axis_sizes:
                continue
            size = max(recorded, planned)
            actual_threads *= max(size, 1)
        if num_threads > actual_threads:
            # Some logical axes are being serialized (for example via lane loops)
            # rather than mapped to actual threads. The strided thread-reduction
            # path assumes every participating lane is backed by a live thread, so
            # using it here would read unwritten SMEM partials.
            debug(
                "skip actual threads",
                tuple(fake_input.size()),
                dim,
                num_threads,
                actual_threads,
                logical_axis_sizes,
            )
            return None
        if pre <= 1 and group_span <= 32 and num_threads == group_span:
            debug(
                "skip small direct",
                tuple(fake_input.size()),
                dim,
                "block",
                self.block_index,
                "reduce_axis",
                reduce_axis,
                "pre",
                pre,
                "group_span",
                group_span,
                "mapping",
                tensor_dim_mapping,
                "active_thread_blocks",
                active_thread_blocks,
                "logical_axis_sizes",
                logical_axis_sizes,
            )
            return None
        debug(
            "use strided",
            tuple(fake_input.size()),
            dim,
            "block",
            self.block_index,
            "reduce_axis",
            reduce_axis,
            "pre",
            pre,
            "group_span",
            group_span,
            "mapping",
            tensor_dim_mapping,
            "active_thread_blocks",
            active_thread_blocks,
            "logical_axis_sizes",
            logical_axis_sizes,
        )
        if group_span > 32:
            assert num_threads % group_span == 0, (
                f"num_threads ({num_threads}) must be divisible by "
                f"group_span ({group_span})"
            )
            smem_budget_bytes = _cute_shared_memory_budget_bytes()
            group_count = num_threads // group_span
            lane_var = self.fn.new_var("strided_lane", dce=True)
            lane_in_group_var = self.fn.new_var("strided_lane_in_group", dce=True)
            lane_mod_pre_var = self.fn.new_var("strided_lane_mod_pre", dce=True)
            state.add_statement(f"{lane_var} = {lane_expr}")
            state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
            state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
            if group_span % 32 == 0:
                warps_per_group = group_span // 32
                partials_size = group_count * pre * warps_per_group
                results_size = group_count * pre
                if (
                    _cute_reduction_smem_bytes(
                        partials_size + results_size, fake_input.dtype
                    )
                    > smem_budget_bytes
                ):
                    return None
                return self._strided_thread_reduction_expr_shared_two_stage(
                    state=state,
                    input_name=input_name,
                    reduction_type=reduction_type,
                    fake_input=fake_input,
                    identity_expr=identity_expr,
                    lane_var=lane_var,
                    lane_in_group_var=lane_in_group_var,
                    lane_mod_pre_var=lane_mod_pre_var,
                    pre=pre,
                    group_span=group_span,
                    group_count=group_count,
                )
            if (
                _cute_reduction_smem_bytes(
                    num_threads + group_count * pre, fake_input.dtype
                )
                > smem_budget_bytes
            ):
                return None
            return self._strided_thread_reduction_expr_shared_tree(
                state=state,
                input_name=input_name,
                reduction_type=reduction_type,
                fake_input=fake_input,
                identity_expr=identity_expr,
                lane_var=lane_var,
                lane_in_group_var=lane_in_group_var,
                lane_mod_pre_var=lane_mod_pre_var,
                pre=pre,
                group_span=group_span,
                num_threads=num_threads,
                group_count=group_count,
            )

        lane_in_group = f"(({lane_expr}) % {group_span})"
        lane_mod_pre = f"(({lane_in_group}) % {pre})"
        reduced_terms: list[str] = []
        for p in range(pre):
            masked_input_var = self.fn.new_var(f"strided_masked_input_{p}", dce=True)
            reduced_var = self.fn.new_var(f"strided_reduced_{p}", dce=True)
            state.add_statement(
                f"{masked_input_var} = ({input_name}) if ({lane_mod_pre}) == {p} else ({identity_expr})"
            )
            # dim=0: reducing scalar per-thread values across warp lanes
            reduction = backend.reduction_expr(
                masked_input_var, reduction_type, 0, threads_in_group=group_span
            )
            state.add_statement(f"{reduced_var} = {reduction}")
            reduced_terms.append(reduced_var)
        selected_result = reduced_terms[0]
        for p, reduced in enumerate(reduced_terms[1:], start=1):
            selected_result = (
                f"({reduced}) if ({lane_mod_pre}) == {p} else ({selected_result})"
            )
        return selected_result

    def _strided_thread_reduction_expr_shared_two_stage(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_input: torch.Tensor,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> str:
        backend = CompileEnvironment.current().backend
        dtype = _dtype_str(fake_input.dtype)
        warps_per_group = group_span // 32
        partials_size = group_count * pre * warps_per_group
        results_size = group_count * pre
        smem_size = partials_size + results_size
        smem_ptr_var = self.fn.new_var("strided_reduce_smem_ptr", dce=True)
        smem_var = self.fn.new_var("strided_reduce_smem", dce=True)
        group_id_var = self.fn.new_var("strided_group_id", dce=True)
        lane_in_warp_var = self.fn.new_var("strided_lane_in_warp", dce=True)
        warp_in_group_var = self.fn.new_var("strided_warp_in_group", dce=True)
        partials_base_var = self.fn.new_var("strided_partials_base", dce=True)
        results_base_var = self.fn.new_var("strided_results_base", dce=True)
        state.add_statement(
            f"{smem_ptr_var} = cute.arch.alloc_smem({dtype}, {smem_size})"
        )
        state.add_statement(
            f"{smem_var} = cute.make_tensor({smem_ptr_var}, ({smem_size},))"
        )
        state.add_statement(f"{group_id_var} = ({lane_var}) // {group_span}")
        state.add_statement(f"{lane_in_warp_var} = ({lane_var}) % 32")
        state.add_statement(f"{warp_in_group_var} = ({lane_in_group_var}) // 32")
        state.add_statement(
            f"{partials_base_var} = ({group_id_var}) * {pre * warps_per_group}"
        )
        state.add_statement(
            f"{results_base_var} = {partials_size} + ({group_id_var}) * {pre}"
        )

        for p in range(pre):
            masked_input_var = self.fn.new_var(f"strided_masked_input_{p}", dce=True)
            warp_partial_var = self.fn.new_var(f"strided_warp_partial_{p}", dce=True)
            partial_idx_var = self.fn.new_var(f"strided_partial_idx_{p}", dce=True)
            stage2_input_var = self.fn.new_var(f"strided_stage2_input_{p}", dce=True)
            group_result_var = self.fn.new_var(f"strided_group_result_{p}", dce=True)
            state.add_statement(
                f"{masked_input_var} = ({input_name}) if ({lane_mod_pre_var}) == {p} else ({identity_expr})"
            )
            state.add_statement(
                f"{warp_partial_var} = {backend.reduction_expr(masked_input_var, reduction_type, 0, threads_in_group=32)}"
            )
            state.add_statement(
                f"{partial_idx_var} = ({partials_base_var}) + {p * warps_per_group} + ({warp_in_group_var})"
            )
            state.add_statement(
                statement_from_string(
                    f"""if ({lane_in_warp_var}) == 0:
    {smem_var}[{partial_idx_var}] = {warp_partial_var}"""
                )
            )
            state.add_statement("cute.arch.sync_threads()")

            state.add_statement(
                statement_from_string(
                    f"""if ({warp_in_group_var}) == 0:
    {stage2_input_var} = {smem_var}[({partials_base_var}) + {p * warps_per_group} + ({lane_in_warp_var})] if ({lane_in_warp_var}) < {warps_per_group} else ({identity_expr})
    {group_result_var} = {backend.reduction_expr(stage2_input_var, reduction_type, 0, threads_in_group=32)}
    if ({lane_in_warp_var}) == 0:
        {smem_var}[({results_base_var}) + {p}] = {group_result_var}"""
                )
            )
            state.add_statement("cute.arch.sync_threads()")

        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = {smem_var}[({results_base_var}) + ({lane_mod_pre_var})]"
        )
        return result_var

    def _strided_thread_reduction_expr_shared_tree(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_input: torch.Tensor,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        num_threads: int,
        group_count: int,
    ) -> str:
        backend = CompileEnvironment.current().backend
        dtype = _dtype_str(fake_input.dtype)
        smem_size = num_threads + group_count * pre
        smem_ptr_var = self.fn.new_var("strided_reduce_smem_ptr", dce=True)
        smem_var = self.fn.new_var("strided_reduce_smem", dce=True)
        group_base_var = self.fn.new_var("strided_group_base", dce=True)
        group_id_var = self.fn.new_var("strided_group_id", dce=True)
        result_base_var = self.fn.new_var("strided_result_base", dce=True)
        state.add_statement(
            f"{smem_ptr_var} = cute.arch.alloc_smem({dtype}, {smem_size})"
        )
        state.add_statement(
            f"{smem_var} = cute.make_tensor({smem_ptr_var}, ({smem_size},))"
        )
        state.add_statement(f"{group_base_var} = ({lane_var}) - ({lane_in_group_var})")
        state.add_statement(f"{group_id_var} = ({lane_var}) // {group_span}")
        state.add_statement(
            f"{result_base_var} = {num_threads} + ({group_id_var}) * {pre}"
        )

        for p in range(pre):
            masked_input_var = self.fn.new_var(f"strided_masked_input_{p}", dce=True)
            state.add_statement(
                f"{masked_input_var} = ({input_name}) if ({lane_mod_pre_var}) == {p} else ({identity_expr})"
            )
            state.add_statement(f"{smem_var}[{lane_var}] = {masked_input_var}")
            state.add_statement("cute.arch.sync_threads()")
            stride = 1
            while stride < group_span:
                cond = (
                    f"(({lane_in_group_var}) % {stride * 2}) == 0"
                    f" and ({lane_in_group_var}) + {stride} < {group_span}"
                )
                lhs = f"{smem_var}[{lane_var}]"
                rhs = (
                    f"{smem_var}[({group_base_var}) + ({lane_in_group_var}) + {stride}]"
                )
                combined = backend.reduction_combine_expr(
                    reduction_type, lhs, rhs, fake_input.dtype
                )
                state.add_statement(
                    statement_from_string(
                        f"""if {cond}:
    {smem_var}[{lane_var}] = {combined}"""
                    )
                )
                state.add_statement("cute.arch.sync_threads()")
                stride *= 2

            state.add_statement(
                statement_from_string(
                    f"""if ({lane_in_group_var}) == 0:
    {smem_var}[({result_base_var}) + {p}] = {smem_var}[{lane_var}]"""
                )
            )
            state.add_statement("cute.arch.sync_threads()")

        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = {smem_var}[({result_base_var}) + ({lane_mod_pre_var})]"
        )
        return result_var

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
        assert isinstance(default, (float, int, bool))
        env = CompileEnvironment.current()
        dim_size = fake_input.size(dim)
        is_zero_dim = False
        if (
            isinstance(dim_size, int)
            and dim_size == 0
            or isinstance(dim_size, torch.SymInt)
            and env.known_equal(dim_size, 0)
        ):
            is_zero_dim = True
        if is_zero_dim:
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                env.backend.full_expr(
                    shape_dims, constant_repr(default), fake_output.dtype
                )
            )
        if (
            strided_expr := self._strided_thread_reduction_expr(
                state, input_name, reduction_type, dim, fake_input, default
            )
        ) is not None:
            expr = strided_expr
        elif env.backend.name == "cute" and self._reduction_block_is_serial():
            # The current reduction block is being traversed by a serial device
            # loop rather than live threads, so the surrounding loop-carried
            # accumulator performs the real reduction. Each iteration should
            # contribute only its current scalar value.
            expr = input_name
        elif env.backend.name == "cute" and self._reduction_block_has_lane_loops():
            # Under active lane loops the reduction is serialized by the
            # surrounding Python loops, so each iteration should contribute its
            # current scalar value directly. Applying a thread reduction here
            # would incorrectly collapse across the live thread lanes instead.
            expr = input_name
        elif (
            env.backend.name == "cute"
            and not self._reduction_block_has_live_thread_axis()
        ):
            # The current reduction block is not backed by a live thread axis
            # in the active loop nest, so reducing across warp lanes would fold
            # together unrelated tensor elements. Let the surrounding loop-
            # carried accumulator perform the reduction instead.
            expr = input_name
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
