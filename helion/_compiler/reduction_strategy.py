from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.codegen.triton import triton_acc_type
from torch._inductor.ir import get_reduction_combine_fn
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import triton_type
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

ARG_REDUCE_MAP = {"argmax": ("max", "maximum"), "argmin": ("min", "minimum")}


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
        if reduction_type in {"sum", "max", "min"}:
            # TODO(jansel): some of the above have different NaN handling than torch, we may want to take the triton_helpers version
            return f"tl.{reduction_type}({input_name}, {dim})"
        if reduction_type in {"argmax", "argmin"}:
            index_var = self.index_var(self.block_index)
            return self.call_argmin_argmax(
                input_name,
                self.broadcast_str(index_var, fake_input, dim),
                reduction_type,
                dim,
                fake_output,
            )
        if reduction_type == "prod":
            return f"triton_helpers.prod({input_name}, {dim})"
        raise NotImplementedError(f"Unsupported reduction type: {reduction_type}")

    def call_argmin_argmax(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
    ) -> str:
        base, _ = ARG_REDUCE_MAP[reduction_type]
        return (
            f"triton_helpers.{base}_with_index("
            f"{input_name}, {index_value}, {dim})[1].to({triton_type(fake_output.dtype)})"
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
        return f"tl.reshape({expr}, {shape})"

    def broadcast_str(self, base: str, fake_input: torch.Tensor, dim: int) -> str:
        input_size = [*fake_input.size()]
        expand = self.fn.tile_strategy.expand_str(input_size, dim)
        shape = self.fn.tile_strategy.shape_str(input_size)
        return f"tl.broadcast_to({base}{expand}, {shape})"


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        numel = CompileEnvironment.current().block_sizes[block_index].numel
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

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
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
                        f"{block_size_var} = triton.next_power_of_2({expr_str})"
                    )
                    state.codegen.host_statements.append(stmt)
        state.add_statement(
            f"{index_var} = tl.arange(0, {block_size_var}).to({env.triton_index_type()})"
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
        state.codegen.set_active_loops(
            PersistentReductionState(self, block_id_to_info=block_id_to_info)
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
                f"{index_var} = {offset_var} + tl.arange(0, ({block_size_var})).to({env.triton_index_type()})"
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
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
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
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            shape = self.fn.tile_strategy.shape_str([*fake_input.size()])
            acc_dtype = get_computation_dtype(fake_input.dtype)  # promote fp16 to fp32
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            assert state.fx_node is not None
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            device_loop.outer_prefix.append(
                statement_from_string(
                    f"{acc} = tl.full({shape}, {constant_repr(default)}, {triton_acc_type(acc_dtype)})"
                )
            )
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if reduction_type not in {"argmin", "argmax"}:
                combine_fn = get_reduction_combine_fn(reduction_type, acc_dtype)
                state.add_statement(f"{acc} = {combine_fn(acc, input_name)}")
                expr = self.call_reduction_function(
                    acc, reduction_type, dim, fake_input, fake_output
                )
            else:
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = CompileEnvironment.current().settings.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = tl.full({shape}, {torch.iinfo(index_dtype).max!r}, {triton_type(index_dtype)})"
                    )
                )
                index = self.broadcast_str(
                    self.index_var(self.block_index), fake_input, dim
                )
                _, combine = ARG_REDUCE_MAP[reduction_type]
                state.add_statement(
                    f"{acc}, {acc_index} = triton_helpers.{combine}_with_index("
                    f"{acc}, {acc_index}, {input_name}, {index})"
                )
                expr = self.call_argmin_argmax(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                )
            # Ensure the final reduction result matches torch.* dtype semantics
            expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
            expr = f"tl.cast({expr}, {triton_type(fake_output.dtype)})"
            device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))

            # Optional: emit a dtype static assert right after the assignment when enabled
            if CompileEnvironment.current().settings.debug_dtype_asserts:
                device_loop.outer_suffix.append(
                    statement_from_string(
                        f"tl.static_assert({result}.dtype == {triton_type(fake_output.dtype)})"
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
        expr = self.call_reduction_function(
            input_name,
            reduction_type,
            dim,
            fake_input,
            fake_output,
        )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
