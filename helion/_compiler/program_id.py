from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.host_function import HostFunction

if TYPE_CHECKING:
    import sympy

    from helion._compiler.inductor_lowering import CodegenState


class ProgramID(NamedTuple):
    pid_var: str
    block_size_var: str
    numel: sympy.Expr

    def host_cdiv(self) -> str:
        numel_str = HostFunction.current().sympy_expr(self.numel)
        if self.block_size_var == "1":
            return numel_str
        return f"triton.cdiv({numel_str}, {self.block_size_var})"

    def device_cdiv(self, state: CodegenState) -> str:
        numel_str = state.device_function.sympy_expr(self.numel)
        if self.block_size_var == "1":
            return numel_str
        return f"tl.cdiv({numel_str}, {self.block_size_var})"


@dataclasses.dataclass
class ProgramIDs:
    pids: list[ProgramID] = dataclasses.field(default_factory=list)

    def append(self, pid: ProgramID) -> None:
        self.pids.append(pid)

    def codegen(self, state: CodegenState) -> None:
        raise NotImplementedError


class GridProgramIDs(ProgramIDs):
    """Use the cuda x/y/z launch grid for PIDs"""

    def codegen(self, state: CodegenState) -> None:
        grid = []
        for i, pid in enumerate(self.pids):
            state.codegen.statements_stack[-1].insert(
                i, statement_from_string(f"{pid.pid_var} = tl.program_id({i})")
            )
            grid.append(pid.host_cdiv())
        assert len(grid) <= 3
        state.device_function.set_grid_expr(expr_from_string(f"({', '.join(grid)},)"))


class VirtualProgramIDs(ProgramIDs):
    """Only use the x grid and compute other dimensions"""

    def codegen(self, state: CodegenState) -> None:
        num_blocks = [
            state.device_function.new_var(f"num_blocks_{i}")
            for i in range(len(self.pids[:-1]))
        ]
        statements = [
            statement_from_string(f"{num_block} = {pid.device_cdiv(state)}")
            for num_block, pid in zip(num_blocks, self.pids[:-1], strict=True)
        ]
        for i, pid in enumerate(self.pids):
            expr = "tl.program_id(0)"
            if i > 0:
                divisor = " * ".join(num_blocks[:i])
                expr = f"({expr}) // ({divisor})"
            if i + 1 < len(self.pids):
                expr = f"({expr}) % ({num_blocks[i]})"
            statements.append(statement_from_string(f"{pid.pid_var} = {expr}"))
        state.codegen.statements_stack[-1][:] = [
            *statements,
            *state.codegen.statements_stack[-1],
        ]
        state.device_function.set_grid_expr(
            expr_from_string(f"({' * '.join(pid.host_cdiv() for pid in self.pids)},)")
        )


@dataclasses.dataclass
class L2GroupingProgramIDs(ProgramIDs):
    """Used grouped iteration order to promote L2 cache reuse in matmuls"""

    group_size: int = 1

    def codegen(self, state: CodegenState) -> None:
        assert len(self.pids) == 2
        new_var = state.device_function.new_var
        pid = "tl.program_id(0)"
        num_pid_m = new_var("num_pid_m")
        num_pid_n = new_var("num_pid_n")
        num_pid_in_group = new_var("num_pid_in_group")
        group_id = new_var("group_id")
        first_pid_m = new_var("first_pid_m")
        group_size_m = new_var("group_size_m")
        state.codegen.statements_stack[-1][:] = [
            statement_from_string(f"{num_pid_m} = {self.pids[0].device_cdiv(state)}"),
            statement_from_string(f"{num_pid_n} = {self.pids[1].device_cdiv(state)}"),
            statement_from_string(
                f"{num_pid_in_group} = {self.group_size} * {num_pid_n}"
            ),
            statement_from_string(f"{group_id} = {pid} // {num_pid_in_group}"),
            statement_from_string(f"{first_pid_m} = {group_id} * {self.group_size}"),
            statement_from_string(
                f"{group_size_m} = min({num_pid_m} - {first_pid_m}, {self.group_size})"
            ),
            statement_from_string(
                f"{self.pids[0].pid_var} = {first_pid_m} + (({pid} % {num_pid_in_group}) % {group_size_m})"
            ),
            statement_from_string(
                f"{self.pids[1].pid_var} = ({pid} % {num_pid_in_group}) // {group_size_m}"
            ),
            *state.codegen.statements_stack[-1],
        ]
        state.device_function.set_grid_expr(
            expr_from_string(f"({' * '.join(pid.host_cdiv() for pid in self.pids)},)")
        )
