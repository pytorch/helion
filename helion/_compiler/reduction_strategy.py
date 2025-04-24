from __future__ import annotations

from typing import TYPE_CHECKING

import sympy

from ..autotuner.config_fragment import integer_power_of_two
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .host_function import HostFunction
from .tile_strategy import CompactedShape
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    import ast

    import torch

    from .inductor_lowering import CodegenState


class ReductionStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=fn,
            block_indices=[block_index],
        )

    @property
    def block_index(self) -> int:
        return self.block_indices[0]

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        return shapes

    def codegen_reduction(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        raise NotImplementedError

    def call_reduction_function(
        self, input_name: str, reduction_type: str, dim: int
    ) -> str:
        if reduction_type in {"sum", "max", "min", "argmax", "argmin"}:
            # TODO(jansel): some of the above have different NaN handling than torch, we may want to take the triton_helpers version
            return f"tl.{reduction_type}({input_name}, {dim})"
        if reduction_type == "prod":
            return f"triton_helpers.prod({input_name}, {dim})"
        raise NotImplementedError(f"Unsupported reduction type: {reduction_type}")


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=fn,
            block_index=block_index,
        )
        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        if isinstance(numel, (int, sympy.Integer)) and integer_power_of_two(int(numel)):
            self._mask_var: str | None = None
        else:
            self._mask_var = self.fn.new_var(f"mask_{block_index}", dce=True)
        self._block_size_var: str = self.fn.new_var(f"_RDIM_SIZE_{block_index}")
        self.offset_vars[block_index] = "0"

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._block_size_var

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        mask_var = self._mask_var
        block_size_var = self._block_size_var
        state.codegen.host_statements.append(
            statement_from_string(
                f"{block_size_var} = triton.next_power_of_2({HostFunction.current().sympy_expr(numel)})"
            )
        )
        state.device_function.constexpr_arg(block_size_var)
        state.add_statement(
            f"{index_var} = tl.arange(0, {block_size_var}).to({env.triton_index_type()})"
        )
        if mask_var is not None:
            state.add_statement(
                f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
            )

    def codegen_reduction(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        expr = self.call_reduction_function(input_name, reduction_type, dim)
        size = [*fake_input.size()]
        size.pop(dim)
        if [*fake_output.size()] == size:
            return expr_from_string(expr)
        shape = DeviceFunction.current().tile_strategy.shape_str([*fake_output.size()])
        return expr_from_string(f"tl.reshape({expr}, {shape})")
