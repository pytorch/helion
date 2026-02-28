from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._prims_common import get_computation_dtype

from ..autotuner.config_fragment import integer_power_of_two
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import install_inductor_kernel_handlers
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceLoopState
from .tile_strategy import PersistentReductionState
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _acc_type(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.acc_type(dtype)


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
                # pyrefly: ignore [no-matching-overload]
                size_hint = int(env.shape_env.size_hint(numel))
            else:
                size_hint = env.size_hint(numel)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
        else:
            self._thread_count = 0

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
        thread_axis_sizes = {}
        block_thread_axes = {}
        axis = self._get_thread_axis()
        if self._thread_count > 0:
            thread_axis_sizes[axis] = self._thread_count
            block_thread_axes[self.block_index] = axis
        state.codegen.set_active_loops(
            PersistentReductionState(
                self,
                block_id_to_info=block_id_to_info,
                thread_axis_sizes=thread_axis_sizes,
                block_thread_axes=block_thread_axes,
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
        thread_axis_sizes = {}
        block_thread_axes = {}
        axis = self._get_thread_axis()
        if self._thread_count > 0:
            thread_axis_sizes[axis] = self._thread_count
            block_thread_axes[block_index] = axis
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=thread_axis_sizes,
            block_thread_axes=block_thread_axes,
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
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
                block_axes.update(loop_state.block_thread_axes)
        return block_axes, axis_sizes

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
        if backend.name != "cute":
            return None
        if backend.is_indexed_reduction(reduction_type):
            return None

        block_axes, axis_sizes = self._active_thread_layout()
        reduce_axis = block_axes.get(self.block_index)
        if reduce_axis is None:
            strategy = self.fn.tile_strategy.block_id_to_strategy.get(
                (self.block_index,)
            )
            if strategy is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduce_axis is not None:
                hint = backend.reduction_threads_hint(
                    self.block_size_var(self.block_index)
                )
                if hint is not None:
                    axis_sizes[reduce_axis] = max(axis_sizes.get(reduce_axis, 1), hint)
        if reduce_axis is None:
            return None

        pre = 1
        for axis in range(reduce_axis):
            pre *= axis_sizes.get(axis, 1)
        if pre <= 1:
            return None

        reduce_extent = axis_sizes.get(reduce_axis, 1)
        group_span = pre * reduce_extent
        lane_expr = backend.thread_linear_index_expr(axis_sizes)
        if lane_expr is None:
            return None

        dtype = _dtype_str(fake_input.dtype)
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype)
        if group_span > 32:
            num_threads = 1
            for size in axis_sizes.values():
                num_threads *= size
            assert num_threads % group_span == 0, (
                f"num_threads ({num_threads}) must be divisible by "
                f"group_span ({group_span})"
            )
            lane_var = self.fn.new_var("strided_lane", dce=True)
            lane_in_group_var = self.fn.new_var("strided_lane_in_group", dce=True)
            lane_mod_pre_var = self.fn.new_var("strided_lane_mod_pre", dce=True)
            state.add_statement(f"{lane_var} = {lane_expr}")
            state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
            state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
            if group_span % 32 == 0:
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
                    group_count=num_threads // group_span,
                )
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
                group_count=num_threads // group_span,
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
            reduction = backend.reduction_expr(
                masked_input_var, reduction_type, dim, threads_in_group=group_span
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
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
