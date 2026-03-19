from __future__ import annotations

import abc
import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .device_function import TensorArg
from .device_function import TensorDescriptorArg
from .host_function import HostFunction


def typed_program_id(dim: int = 0) -> str:
    """Generate backend-specific program ID expression.

    Triton uses tl.program_id(). CuTe uses block_idx() as the virtual program ID.
    """
    env = CompileEnvironment.current()
    return env.backend.program_id_expr(dim, index_dtype=env.index_type())


if TYPE_CHECKING:
    import sympy

    from .inductor_lowering import CodegenState

NUM_SM_VAR = "_NUM_SM"


class PIDInfo(NamedTuple):
    pid_var: str
    block_size_var: str
    numel: sympy.Expr | str  # Can be a sympy.Expr or a string for data-dependent bounds
    block_id: int

    def num_pids_expr(self, *, is_device: bool) -> str:
        """Get the number of PIDs expression for device or host."""
        if is_device:
            context = DeviceFunction.current()
        else:
            context = HostFunction.current()
        # Handle both sympy.Expr and string numel (for data-dependent bounds)
        if isinstance(self.numel, str):
            numel_str = self.numel
        else:
            numel_str = context.sympy_expr(self.numel)
        if self.block_size_var == "1":
            return numel_str
        return CompileEnvironment.current().backend.cdiv_expr(
            numel_str, self.block_size_var, is_device=is_device
        )


@dataclasses.dataclass
class ProgramIDs(abc.ABC):
    """Base class for all program ID strategies with common functionality."""

    shared_pid_var: str | None = None
    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list)

    def append(self, pid: PIDInfo) -> None:
        self.pid_info.append(pid)

    @abc.abstractmethod
    def codegen(self, state: CodegenState) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def codegen_grid(self) -> ast.AST:
        """Generate grid launch expression for kernel execution."""
        raise NotImplementedError

    def total_pids_expr(self, *, is_device: bool) -> str:
        """Get total PIDs expression for device or host."""
        return " * ".join(
            f"({pid.num_pids_expr(is_device=is_device)})" for pid in self.pid_info
        )

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Setup persistent kernel if supported. Returns None if not a persistent kernel."""
        return None

    def _setup_persistent_kernel_and_wrap_body(
        self,
        device_function: DeviceFunction,
        virtual_pid_var: str,
        range_expr: str,
        total_pids_expr: str | None = None,
    ) -> list[ast.stmt]:
        """Complete persistent kernel setup: prepare body, wrap in loop, and return."""
        from .ast_extension import create

        # Prepare body for persistent loop
        wrapped_body = list(device_function.body)
        if isinstance(device_function.pid, ForEachProgramID):
            shared_pid_var = device_function.pid.shared_pid_var
            wrapped_body = [
                statement_from_string(f"{shared_pid_var} = {virtual_pid_var}"),
                *wrapped_body,
            ]

        # Create the persistent loop that wraps the entire body
        persistent_loop = create(
            ast.For,
            target=create(ast.Name, id=virtual_pid_var, ctx=ast.Store()),
            iter=expr_from_string(range_expr),
            body=wrapped_body,
            orelse=[],
            type_comment=None,
        )
        return [persistent_loop]

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression for this strategy."""
        return typed_program_id(0)

    def _is_persistent(self) -> bool:
        """Check if this is a persistent strategy. Default False."""
        return False

    def _decompose_pid_to_statements(
        self, pid_var: str, state: CodegenState
    ) -> list[ast.stmt]:
        """Generate statements to decompose a single PID variable into multiple PID components."""
        num_blocks = [
            state.device_function.new_var(f"num_blocks_{i}")
            for i in range(len(self.pid_info[:-1]))
        ]
        statements = [
            statement_from_string(f"{num_block} = {pid.num_pids_expr(is_device=True)}")
            for num_block, pid in zip(num_blocks, self.pid_info[:-1], strict=True)
        ]
        for i, pid in enumerate(self.pid_info):
            expr = pid_var
            if i > 0:
                divisor = " * ".join(num_blocks[:i])
                expr = f"({expr}) // ({divisor})"
            if i + 1 < len(self.pid_info):
                expr = f"({expr}) % ({num_blocks[i]})"
            statements.append(statement_from_string(f"{pid.pid_var} = {expr}"))
        return statements


@dataclasses.dataclass
class ForEachProgramID(ProgramIDs):
    """
    Represent multiple top level for loops in the Helion kernel.  Turns into `if` statements in generated code.
    """

    # pyrefly: ignore [bad-override]
    shared_pid_var: str
    cases: list[ProgramIDs] = dataclasses.field(default_factory=list)
    case_phases: list[int] = dataclasses.field(default_factory=list)
    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list, init=False)
    barrier_after_root: set[int] = dataclasses.field(default_factory=set)

    def codegen_pid_init(self) -> list[ast.stmt]:
        # Check if persistent kernels are enabled in config - if so, skip regular initialization
        # as it will be handled by the persistent loop wrapper
        from .device_function import DeviceFunction

        current_device_fn = DeviceFunction.current()
        pid_type = current_device_fn.config.get("pid_type", "flat")
        if isinstance(pid_type, str) and pid_type.startswith("persistent"):
            return []
        return [statement_from_string(f"{self.shared_pid_var} = {typed_program_id(0)}")]

    def _get_cdiv_blocks(
        self, state: CodegenState, exclude_last: bool = False
    ) -> list[str]:
        """Get non-empty cdiv expressions from cases."""
        cases = self.cases[:-1] if exclude_last else self.cases
        blocks = []
        for pid in cases:
            cdiv = pid.total_pids_expr(is_device=True)
            if cdiv:  # Only add non-empty cdiv expressions
                blocks.append(cdiv)
        return blocks

    def codegen_test(self, state: CodegenState) -> ast.AST:
        blocks = self._get_cdiv_blocks(state)
        return expr_from_string(f"{self.shared_pid_var} < ({'+ '.join(blocks)})")

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        total_expr = self.total_pids_expr(is_device=True)
        # If there is only one phase, fall back to existing behavior.
        has_phases = len(set(self.case_phases)) > 1

        def _base_strategy(pid: ProgramIDs) -> ProgramIDs:
            from .tile_strategy import L2GroupingProgramIDs

            if isinstance(pid, L2GroupingProgramIDs):
                assert pid.parent_strategy is not None, (
                    "L2 grouping strategy is missing its parent"
                )
                return pid.parent_strategy
            return pid

        base_strategy = _base_strategy(self.cases[0])

        if not has_phases:
            return base_strategy.setup_persistent_kernel(device_function, total_expr)

        # We expect a persistent-blocked strategy when barriers are present.
        if not base_strategy._is_persistent():
            return base_strategy.setup_persistent_kernel(device_function, total_expr)

        assert isinstance(base_strategy, PersistentProgramIDs)
        assert base_strategy.is_blocked, (
            "hl.barrier() currently requires persistent_blocked"
        )

        # Delegate to helper for phase-split persistent loops
        return self._emit_phase_loops(base_strategy, device_function, total_expr)

    def total_pids_expr(self, *, is_device: bool) -> str:
        """Get total PIDs expression for ForEachProgramID (sum of all pids)."""
        cdivs = [pid.total_pids_expr(is_device=is_device) for pid in self.cases]
        return " + ".join(cdivs)

    def codegen(self, state: CodegenState) -> None:
        blocks = self._get_cdiv_blocks(state, exclude_last=True)
        if blocks:
            state.codegen.statements_stack[-1].insert(
                0,
                statement_from_string(
                    f"{self.shared_pid_var} -= ({'+ '.join(blocks)})"
                ),
            )

    def codegen_grid(self) -> ast.AST:
        # Check if any of the pids is a persistent strategy
        if self.cases[0]._is_persistent():
            # Use SM count grid for persistent kernels
            return self.cases[0].codegen_grid()

        # When persistent kernels are not active, use the full grid size
        host_cdivs = [pid.total_pids_expr(is_device=False) for pid in self.cases]
        return expr_from_string(f"({'+ '.join(host_cdivs)},)")

    def _prepare_persistent_body(
        self,
        body: list[ast.AST],
        device_function: DeviceFunction,
        virtual_pid_var: str,
    ) -> list[ast.AST]:
        """Prepare body for persistent loop - handle ForEachProgramID assignment."""
        # In persistent kernels, replace ForEachProgramID init with virtual_pid assignment
        return [
            statement_from_string(f"{self.shared_pid_var} = {virtual_pid_var}"),
            *body,
        ]

    def _phase_boundaries(self) -> list[str]:
        """Compute cumulative PID boundaries at phase transitions."""
        cdivs = [pid.total_pids_expr(is_device=True) for pid in self.cases]
        boundaries: list[str] = []
        running = "0"
        prev_phase = self.case_phases[0]
        for idx, cdiv in enumerate(cdivs):
            running = f"({running}) + ({cdiv})"
            next_phase = (
                self.case_phases[idx + 1]
                if idx + 1 < len(self.case_phases)
                else prev_phase
            )
            if next_phase != prev_phase or idx == len(cdivs) - 1:
                boundaries.append(running)
            prev_phase = next_phase
        return boundaries

    def _emit_phase_loops(
        self,
        strategy: PersistentProgramIDs,
        device_function: DeviceFunction,
        total_expr: str,
    ) -> list[ast.stmt]:
        """Emit persistent loops split by KernelPhase boundaries."""
        from .tile_strategy import TileStrategy

        backend = CompileEnvironment.current().backend
        device_function.preamble.extend(
            strategy._persistent_setup_statements(total_expr)
        )

        boundaries = self._phase_boundaries()
        block_ids = [pid.block_id for pid in strategy.pid_info]

        def range_expr(begin: str, end: str) -> str:
            return TileStrategy.get_range_call_str(
                device_function.config, block_ids, begin=begin, end=end
            )

        base_body = self._prepare_persistent_body(
            device_function.body, device_function, strategy.virtual_pid_var
        )

        barrier_stmt = None
        if len(boundaries) > 1:
            sem_arg = device_function.new_var("x_grid_sem", dce=False)
            barrier_stmt = backend.grid_barrier_stmt(sem_arg)
            if barrier_stmt is not None:
                barrier_dtype = backend.barrier_semaphore_dtype()
                device_function.arguments.append(
                    TensorArg(
                        sem_arg,
                        torch.empty(1, device="meta", dtype=barrier_dtype),
                        f"torch.zeros((1,), device={strategy.get_device_str()}, dtype={barrier_dtype})",
                    )
                )

        loops: list[ast.stmt] = []
        start_expr = "0"
        for boundary in boundaries:
            cond = expr_from_string(
                f"({strategy.virtual_pid_var} >= ({start_expr})) and ({strategy.virtual_pid_var} < ({boundary}))"
            )
            loop_body = [create(ast.If, test=cond, body=list(base_body), orelse=[])]
            loops.append(
                create(
                    ast.For,
                    target=create(
                        ast.Name, id=strategy.virtual_pid_var, ctx=ast.Store()
                    ),
                    iter=expr_from_string(
                        range_expr(strategy.start_pid_var, strategy.end_pid_var)
                    ),
                    body=loop_body,
                    orelse=[],
                    type_comment=None,
                )
            )
            if boundary != boundaries[-1] and barrier_stmt is not None:
                loops.append(statement_from_string(barrier_stmt))
            start_expr = boundary
        return loops


class XYZProgramIDs(ProgramIDs):
    """Use the cuda x/y/z launch grid for PIDs"""

    def codegen(self, state: CodegenState) -> None:
        for i, pid in enumerate(self.pid_info):
            state.codegen.statements_stack[-1].insert(
                i, statement_from_string(f"{pid.pid_var} = {typed_program_id(i)}")
            )

    def codegen_grid(self) -> ast.AST:
        env = CompileEnvironment.current()
        if env.backend.name != "pallas":
            assert len(self.pid_info) <= 3
        return expr_from_string(
            f"({', '.join(pid.num_pids_expr(is_device=False) for pid in self.pid_info)},)"
        )

    @property
    def virtual_program_id(self) -> str:
        """
        XYZProgramIDs uses multi-dimensional program IDs and doesn't have a single
        virtual program ID. Wrappers like L2GroupingProgramIDs must explicitly
        handle XYZProgramIDs by flattening the multi-dimensional IDs themselves.
        """
        raise NotImplementedError(
            "XYZProgramIDs does not support virtual_program_id. "
            "Use explicit flattening of multi-dimensional program IDs instead."
        )


class FlatProgramIDs(ProgramIDs):
    """Only use the x grid and compute other dimensions"""

    def codegen(self, state: CodegenState) -> None:
        pid_var = self.shared_pid_var or typed_program_id(0)
        statements = self._decompose_pid_to_statements(pid_var, state)
        state.codegen.statements_stack[-1][:] = [
            *statements,
            *state.codegen.statements_stack[-1],
        ]

    def codegen_grid(self) -> ast.AST:
        return expr_from_string(f"({self.total_pids_expr(is_device=False)},)")


class CuteProgramIDs(FlatProgramIDs):
    """Flat PID strategy for CuTe pointwise kernels."""


@dataclasses.dataclass
class L2GroupingProgramIDs(ProgramIDs):
    """Used grouped iteration order to promote L2 cache reuse in matmuls"""

    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list, init=False)
    parent_strategy: ProgramIDs | None = dataclasses.field(default=None)
    group_size: int = 1

    def append(self, pid: PIDInfo) -> None:
        """Delegate to parent strategy."""
        assert self.parent_strategy is not None
        self.parent_strategy.append(pid)

    def codegen(self, state: CodegenState) -> None:
        # Generate L2 grouping logic
        # Note: Persistent kernel setup is handled by ForEachProgramID if needed
        assert self.parent_strategy is not None
        parent_pids = self.parent_strategy.pid_info
        assert len(parent_pids) >= 2, "L2 grouping requires at least 2 dimensions"
        new_var = state.device_function.new_var

        # Apply L2 grouping to the 2 fastest varying dimensions (pid_0, pid_1)
        # These are always the first 2 dimensions in the PID decomposition
        num_dims = len(parent_pids)
        assignments = []

        # Generate size variables for all dimensions (except the last which doesn't need one)
        num_blocks: list[str] = []
        for i in range(num_dims - 1):
            num_block_var = new_var(f"num_blocks_{i}", dce=True)
            assignments.append(
                (num_block_var, parent_pids[i].num_pids_expr(is_device=True))
            )
            num_blocks.append(num_block_var)

        # Determine the base PID to use for L2 grouping.
        # For XYZ strategy, we need to compute a flattened index from the multi-dimensional
        # program IDs since L2 grouping works on a flat 1D PID space.
        if isinstance(self.parent_strategy, XYZProgramIDs):
            # XYZ uses separate program_id(0), program_id(1), etc. for each dimension.
            # We flatten these into a single index using row-major order:
            # flattened_pid = pid_0 + pid_1 * num_blocks_0 + pid_2 * num_blocks_0 * num_blocks_1 + ...
            terms = [typed_program_id(0)]
            for i in range(1, num_dims):
                multiplier = " * ".join(num_blocks[:i])
                terms.append(f"{typed_program_id(i)} * ({multiplier})")
            pid = " + ".join(terms)
        elif isinstance(state.device_function.pid, ForEachProgramID):
            # For ForEachProgramID, use the shared PID variable
            pid = state.device_function.pid.shared_pid_var
        else:
            # For other strategies (Flat, Persistent), use the virtual_program_id
            pid = self.virtual_program_id

        # Apply L2 grouping to the 2 fastest varying dimensions (pid_0, pid_1)
        fastest_m_idx = 0  # pid_0 (fastest varying)
        fastest_n_idx = 1  # pid_1 (second fastest varying)

        # Extract the 2D portion for the fastest 2 dimensions
        inner_2d_size = new_var("inner_2d_size", dce=True)
        inner_2d_pid = new_var("inner_2d_pid", dce=True)

        num_pid_m = new_var("num_pid_m", dce=True)
        num_pid_n = new_var("num_pid_n", dce=True)
        num_pid_in_group = new_var("num_pid_in_group", dce=True)
        group_id = new_var("group_id", dce=True)
        first_pid_m = new_var("first_pid_m", dce=True)
        group_size_m = new_var("group_size_m", dce=True)

        # Set up L2 grouping for the fastest 2 dimensions
        inner_2d_assignments = [
            (num_pid_m, parent_pids[fastest_m_idx].num_pids_expr(is_device=True)),
            (num_pid_n, parent_pids[fastest_n_idx].num_pids_expr(is_device=True)),
        ]

        # Only add modulo for 3D+ cases where we need to extract the 2D portion
        if num_dims > 2:
            inner_2d_assignments.extend(
                [
                    (inner_2d_size, f"{num_pid_m} * {num_pid_n}"),
                    (
                        inner_2d_pid,
                        f"{pid} % {inner_2d_size}",
                    ),  # Extract fastest 2D portion
                ]
            )
        else:
            # For 2D case, the entire PID space is the 2D space
            inner_2d_assignments.append((inner_2d_pid, pid))

        assignments.extend(inner_2d_assignments)
        assignments.extend(
            [
                (num_pid_in_group, f"{self.group_size} * {num_pid_n}"),
                (group_id, f"{inner_2d_pid} // {num_pid_in_group}"),
                (first_pid_m, f"{group_id} * {self.group_size}"),
                (group_size_m, f"min({num_pid_m} - {first_pid_m}, {self.group_size})"),
                (
                    parent_pids[fastest_m_idx].pid_var,
                    f"{first_pid_m} + (({inner_2d_pid} % {num_pid_in_group}) % {group_size_m})",
                ),
                (
                    parent_pids[fastest_n_idx].pid_var,
                    f"({inner_2d_pid} % {num_pid_in_group}) // {group_size_m}",
                ),
            ]
        )

        # Process remaining dimensions (if any) using standard decomposition
        for i in range(2, num_dims):
            expr = pid
            # Add divisor for all faster dimensions
            if i > 0:
                divisor = " * ".join(num_blocks[:i])
                expr = f"({expr}) // ({divisor})"
            # Add modulo unless this is the outermost dimension
            if i + 1 < num_dims:  # Not the outermost dimension
                expr = f"({expr}) % {num_blocks[i]}"

            assignments.append((parent_pids[i].pid_var, expr))

        statements = [
            statement_from_string(f"{var} = {expr}") for var, expr in assignments
        ]

        state.codegen.statements_stack[-1][:] = [
            *statements,
            *state.codegen.statements_stack[-1],
        ]

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression using parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.virtual_program_id

    def codegen_grid(self) -> ast.AST:
        assert self.parent_strategy is not None
        return self.parent_strategy.codegen_grid()

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Delegate to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.setup_persistent_kernel(
            device_function, total_pids_expr
        )

    def _is_persistent(self) -> bool:
        """Forward to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy._is_persistent()

    def total_pids_expr(self, *, is_device: bool) -> str:
        """Forward to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.total_pids_expr(is_device=is_device)


class PersistentProgramIDs(ProgramIDs):
    """Base class for persistent kernels that use num_sms grid size."""

    def __init__(self, is_blocked: bool = False) -> None:
        super().__init__()
        self.is_blocked: bool = is_blocked
        device_function = DeviceFunction.current()
        self.virtual_pid_var: str = device_function.new_var("virtual_pid")
        self.total_pids_var: str = device_function.new_var("total_pids")
        # Get num_sm_multiplier from config for multi-occupancy support
        # pyrefly: ignore [bad-assignment]
        self.num_sm_multiplier: int = device_function.config.get("num_sm_multiplier", 1)
        # Compute grid size expression based on multiplier
        if self.num_sm_multiplier == 1:
            self.grid_size_expr: str = NUM_SM_VAR
        else:
            self.grid_size_expr = f"({NUM_SM_VAR} * {self.num_sm_multiplier})"
        # Generate variables and range expression based on strategy type
        if self.is_blocked:
            self.block_size_var: str = device_function.new_var("block_size")
            self.start_pid_var: str = device_function.new_var("start_pid")
            self.end_pid_var: str = device_function.new_var("end_pid")
            self.range_kwargs: dict[str, str] = {
                "begin": self.start_pid_var,
                "end": self.end_pid_var,
            }
        else:
            self.range_kwargs: dict[str, str] = {
                "begin": typed_program_id(0),
                "end": self.total_pids_var,
                "step": self.grid_size_expr,
            }
        if device_function.constexpr_arg(NUM_SM_VAR):
            reserved_sms = CompileEnvironment.current().settings.persistent_reserved_sms
            reserved_arg = f", reserved_sms={reserved_sms}" if reserved_sms > 0 else ""
            device_function.codegen.host_statements.append(
                statement_from_string(
                    f"{NUM_SM_VAR} = helion.runtime.get_num_sm({self.get_device_str()}{reserved_arg})"
                )
            )

    def get_device_str(self) -> str:
        """Get the device string for the current device, reusing the first tensor's origin."""
        host_function = HostFunction.current()
        device = CompileEnvironment.current().device
        origins = [
            o for t, o in host_function.tensor_to_origin.items() if t.device == device
        ]
        if origins:
            return f"{origins[0].host_str()}.device"
        return f"torch.{device!r}"

    def codegen_grid(self) -> ast.AST:
        # Use num_sms * multiplier for persistent kernels (multi-occupancy)
        return expr_from_string(f"({self.grid_size_expr},)")

    def _persistent_setup_statements(self, total_pids_expr: str) -> list[ast.stmt]:
        """Generate the preamble statements for persistent kernel setup."""
        env = CompileEnvironment.current()
        backend = env.backend
        # Cast total_pids to match the index type so all persistent scheduling
        # variables (start_pid, end_pid, etc.) have consistent types.
        if env.index_dtype != torch.int32:
            total_pids_expr = backend.cast_expr(total_pids_expr, env.index_type())
        stmts: list[ast.stmt] = [
            statement_from_string(f"{self.total_pids_var} = {total_pids_expr}"),
        ]
        if (
            self.is_blocked
            and self.block_size_var
            and self.start_pid_var
            and self.end_pid_var
        ):
            stmts.extend(
                [
                    statement_from_string(
                        f"{self.block_size_var} = {backend.cdiv_expr(self.total_pids_var, self.grid_size_expr, is_device=True)}"
                    ),
                    statement_from_string(
                        f"{self.start_pid_var} = {typed_program_id(0)} * {self.block_size_var}"
                    ),
                    statement_from_string(
                        f"{self.end_pid_var} = {self.start_pid_var} + {self.block_size_var}"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{self.end_pid_var} > {self.total_pids_var}"
                        ),
                        body=[
                            statement_from_string(
                                f"{self.end_pid_var} = {self.total_pids_var}"
                            )
                        ],
                        orelse=[],
                    ),
                ]
            )
        return stmts

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Setup persistent kernel and return the wrapped body."""
        # Get total PIDs expression
        if total_pids_expr is None:
            total_pids_expr = self.total_pids_expr(is_device=True)

        device_function.preamble.extend(
            self._persistent_setup_statements(total_pids_expr)
        )
        # Collect all block IDs from PID info for range configuration
        pid_block_ids = []
        for pid_info in self.pid_info:
            pid_block_ids.append(pid_info.block_id)

        from .tile_strategy import TileStrategy

        range_expr = TileStrategy.get_range_call_str(
            device_function.config, pid_block_ids, **self.range_kwargs
        )
        return self._setup_persistent_kernel_and_wrap_body(
            device_function, self.virtual_pid_var, range_expr, total_pids_expr
        )

    def _is_persistent(self) -> bool:
        """Check if this is a persistent strategy."""
        return True

    def _decompose_virtual_pid(
        self,
        state: CodegenState,
        virtual_pid_var: str,
        setup_statements: list[ast.stmt],
    ) -> None:
        """Decompose virtual PID into individual PID variables."""
        # Use shared_pid_var if available, otherwise virtual_pid_var
        pid_var = self.shared_pid_var or virtual_pid_var
        statements = self._decompose_pid_to_statements(pid_var, state)
        setup_statements.extend(statements)

    def _generate_pid_statements(self, state: CodegenState) -> list[ast.stmt]:
        """Generate PID decomposition statements based on setup state."""
        if not self.virtual_pid_var:
            # Generate regular PID decomposition
            return self._decompose_pid_to_statements(
                self.shared_pid_var or typed_program_id(0), state
            )

        # Generate persistent PID decomposition
        statements = []
        self._decompose_virtual_pid(state, self.virtual_pid_var, statements)
        return statements

    def _prepend_statements(
        self, state: CodegenState, statements: list[ast.stmt]
    ) -> None:
        """Prepend statements to current statement stack."""
        current_statements = state.codegen.statements_stack[-1]
        current_statements[:] = [*statements, *current_statements]

    def codegen(self, state: CodegenState) -> None:
        """Common codegen logic for persistent kernels."""
        is_shared_pid = isinstance(state.device_function.pid, ForEachProgramID)

        # Set up persistent loop if needed (non-ForEachProgramID case only)
        if not is_shared_pid and not self.virtual_pid_var:
            self.setup_persistent_kernel(state.device_function)

        # Generate and prepend PID decomposition statements
        statements = self._generate_pid_statements(state)
        self._prepend_statements(state, statements)

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression for persistent strategies."""
        return self.virtual_pid_var


class PersistentBlockedProgramIDs(PersistentProgramIDs):
    """Persistent kernels where each SM processes a contiguous block of virtual PIDs."""

    def __init__(self) -> None:
        super().__init__(is_blocked=True)


class PersistentInterleavedProgramIDs(PersistentProgramIDs):
    """Persistent kernels where each SM processes every num_sms-th virtual PID."""

    def __init__(self) -> None:
        super().__init__(is_blocked=False)


class PersistentJaggedProgramIDs(PersistentProgramIDs):
    """Persistent kernel that flattens group, jagged-M, and sibling-N tiles into one virtual PID loop.

    Instead of iterating over groups in the grid and M/N tiles in serial inner
    loops, this strategy computes a host-side prefix sum of tile counts per group
    and distributes ALL tiles across SMs via a single persistent loop.
    """

    def __init__(self) -> None:
        super().__init__(is_blocked=False)

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        env = CompileEnvironment.current()

        # --- 3a. Identify jagged structure ---
        jagged_map = env.jagged_tile_parent_id
        if not jagged_map:
            raise ValueError("persistent_jagged requires at least one hl.jagged_tile()")
        if len(jagged_map) != 1:
            raise ValueError(
                "persistent_jagged currently supports exactly one hl.jagged_tile()"
            )

        jagged_block_id, parent_block_id = next(iter(jagged_map.items()))

        # Find the inner (sibling) block_ids that are device loops (not grid, not reduction, not jagged)
        # These are tiles that appear as inner for-loops alongside the jagged tile
        sibling_block_ids = self._find_sibling_block_ids(
            device_function, jagged_block_id, parent_block_id
        )

        # Validate: group block_size must be 1
        parent_bs = env.block_sizes[parent_block_id].from_config(device_function.config)
        if parent_bs != 1:
            raise ValueError(
                f"persistent_jagged requires group block_size=1, got {parent_bs}"
            )

        # --- 3b. Inject host-side prefix sum ---
        self._inject_prefix_sum(
            device_function, env, jagged_block_id, parent_block_id, sibling_block_ids
        )

        # --- 3c. Transform the device body ---
        return self._transform_body(
            device_function, env, jagged_block_id, parent_block_id, sibling_block_ids
        )

    def _find_sibling_block_ids(
        self,
        device_function: DeviceFunction,
        jagged_block_id: int,
        parent_block_id: int,
    ) -> list[int]:
        """Find the immediate sibling tile block_ids of the jagged tile.

        These are for-loops found directly inside the jagged for-loop body
        (one level deep only). Deeper-nested loops (e.g. reduction K loops)
        are kept as regular for-loops.
        """
        # First, find the jagged for-loop in the body
        # pyrefly: ignore [bad-argument-type]
        jagged_for = self._find_for_node(device_function.body, jagged_block_id)
        if jagged_for is None:
            return []

        # Collect immediate for-loop children from the jagged body
        return self._collect_immediate_for_block_ids(jagged_for.body)

    def _inject_prefix_sum(
        self,
        device_function: DeviceFunction,
        env: CompileEnvironment,
        jagged_block_id: int,
        parent_block_id: int,
        sibling_block_ids: list[int],
    ) -> None:
        """Add host-side prefix sum computation and kernel args."""
        # Get host-side references to group_offsets, G, N, and block sizes
        # The parent block_id's numel is G (number of groups)
        parent_info = env.block_sizes[parent_block_id]
        host_fn = HostFunction.current()
        g_expr = host_fn.sympy_expr(parent_info.numel)

        # Get the group_offsets tensor name from device_function arguments
        # It's the tensor that was used to compute starts/ends
        # We find it by looking for the tensor arg that produced the jagged parent
        group_offsets_name = self._find_group_offsets_arg(device_function)

        # Block size variables for jagged and sibling dims
        jagged_bs_var = device_function.block_size_var_cache.get(
            (jagged_block_id,), f"_BLOCK_SIZE_{jagged_block_id}"
        )

        # Build the product of sibling tile counts: ceil(N / BS_N) * ...
        sibling_tiles_expr_parts = []
        for sib_id in sibling_block_ids:
            sib_info = env.block_sizes[sib_id]
            sib_numel = host_fn.sympy_expr(sib_info.numel)
            sib_bs_var = device_function.block_size_var_cache.get(
                (sib_id,), f"_BLOCK_SIZE_{sib_id}"
            )
            sibling_tiles_expr_parts.append(
                f"(({sib_numel} + {sib_bs_var} - 1) // {sib_bs_var})"
            )

        sibling_tiles_expr = (
            " * ".join(sibling_tiles_expr_parts) if sibling_tiles_expr_parts else "1"
        )

        prefix_var = "_group_tile_prefix"
        total_tiles_var = "_TOTAL_TILES"

        device_str = self.get_device_str()

        # Host-side prefix sum computation
        host_stmts = [
            f"{prefix_var} = torch.empty({g_expr} + 1, dtype=torch.int64, device={device_str})",
            f"{prefix_var}[0] = 0",
            (
                f"for _g in range({g_expr}):\n"
                f"    _m_g = int({group_offsets_name}[_g + 1].item()) - int({group_offsets_name}[_g].item())\n"
                f"    {prefix_var}[_g + 1] = {prefix_var}[_g] + ((_m_g + {jagged_bs_var} - 1) // {jagged_bs_var}) * {sibling_tiles_expr}"
            ),
            f"{total_tiles_var} = int({prefix_var}[-1].item())",
        ]

        for stmt_str in host_stmts:
            device_function.codegen.host_statements.append(
                statement_from_string(stmt_str)
            )

        # Add _TOTAL_TILES as constexpr kernel arg
        device_function.constexpr_arg(total_tiles_var, total_tiles_var)

        # Add _group_tile_prefix as tensor arg
        fake_prefix = torch.empty(1, dtype=torch.int64, device="meta")
        prefix_arg = TensorArg(prefix_var, fake_prefix, prefix_var)
        device_function.arguments.append(prefix_arg)

    def _find_group_offsets_arg(self, device_function: DeviceFunction) -> str:
        """Find the host-side name of the group_offsets tensor argument.

        Look for a tensor arg whose name suggests it's the offsets tensor.
        We search through existing tensor arguments for the one used in the
        jagged tile computation.
        """
        # The group_offsets tensor is typically the first 1D integer tensor arg
        # that has "offsets" in its host_str or is a 1D int tensor
        for arg in device_function.arguments:
            if isinstance(arg, TensorArg) and not isinstance(arg, TensorDescriptorArg):
                if arg.fake_value.ndim == 1 and arg.fake_value.dtype in (
                    torch.int32,
                    torch.int64,
                ):
                    return arg.host_str()
        # Fallback: look through all tensor args
        for arg in device_function.arguments:
            if isinstance(arg, TensorArg):
                if "offset" in arg.name.lower() or "offset" in arg.host_str().lower():
                    return arg.host_str()
        raise ValueError(
            "Could not find group_offsets tensor argument for persistent_jagged"
        )

    def _transform_body(
        self,
        device_function: DeviceFunction,
        env: CompileEnvironment,
        jagged_block_id: int,
        parent_block_id: int,
        sibling_block_ids: list[int],
    ) -> list[ast.stmt]:
        """Transform the device body to decode virtual_pid and flatten inner loops."""
        body = list(device_function.body)
        prefix_var = "_group_tile_prefix"
        total_tiles_var = "_TOTAL_TILES"

        # Get block size vars
        jagged_bs_var = device_function.block_size_var_cache.get(
            (jagged_block_id,), f"_BLOCK_SIZE_{jagged_block_id}"
        )

        g_expr = HostFunction.current().sympy_expr(
            env.block_sizes[parent_block_id].numel
        )

        # Find the first ast.For node in body (the jagged M loop)
        pre_loop_stmts: list[ast.stmt] = []
        jagged_for: ast.For | None = None

        for stmt in body:
            if isinstance(stmt, ast.For):
                target = stmt.target
                if (
                    isinstance(target, ast.Name)
                    and target.id == f"offset_{jagged_block_id}"
                ):
                    jagged_for = stmt
                    break
            pre_loop_stmts.append(stmt)  # pyrefly: ignore [bad-argument-type]

        if jagged_for is None:
            raise ValueError("Could not find jagged tile for-loop in device body")

        # Extract the jagged for-loop body:
        # It contains setup stmts (indices, mask) + user code + possibly nested for-loops
        jagged_body = list(jagged_for.body)

        # Build the new flattened body:
        # 1. Persistent preamble (total_pids, etc.)
        device_function.preamble.extend(
            self._persistent_setup_statements(total_tiles_var)
        )

        # 2. Decode virtual_pid: binary search over prefix sum
        virtual_pid_var = self.virtual_pid_var
        group_idx_var = device_function.new_var("group_idx")
        local_tile_var = device_function.new_var("local_tile")

        decode_stmts: list[ast.stmt] = [
            statement_from_string(f"{group_idx_var} = 0"),
            # Linear scan: for _scan in tl.range(0, G): if prefix[_scan+1] <= vpid: group_idx = _scan+1
            create(
                ast.For,
                target=create(ast.Name, id="_scan", ctx=ast.Store()),
                iter=expr_from_string(f"tl.range(0, {g_expr})"),
                body=[
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"tl.load({prefix_var} + _scan + 1) <= {virtual_pid_var}"
                        ),
                        body=[
                            statement_from_string(f"{group_idx_var} = _scan + 1"),
                        ],
                        orelse=[],
                    ),
                ],
                orelse=[],
                type_comment=None,
            ),
            statement_from_string(
                f"{local_tile_var} = {virtual_pid_var} - tl.load({prefix_var} + {group_idx_var})"
            ),
        ]

        # 3. Replace pid_0/offset_0 assignment: set offset_0 = group_idx
        # Find the pid/offset assignments in pre_loop_stmts and replace them
        new_pre_loop = self._replace_grid_pid_with_group_idx(
            pre_loop_stmts, group_idx_var
        )

        # 4. After user code (which computes starts, M_g etc.), compute M tiles and set offsets
        # We need to find the variable that holds M_g (the jagged lengths) — it's the variable
        # used in the amax computation. For block_size=1, it's directly accessed.
        # m_tiles = cdiv(M_g_scalar, BS_M) where M_g is per-group
        # But since group_bs=1, M_g is a scalar

        # Build the offset computation for jagged and sibling dims
        m_tiles_var = device_function.new_var("_m_tiles")

        # Compute number of M tiles from the group's M size
        # We need to find the variable that holds M_g. It's computed as ends - starts.
        # Instead of trying to find it, we can compute it from group_offsets directly:
        # m_size = group_offsets[group_idx + 1] - group_offsets[group_idx]
        # But we already have starts and ends in the pre_loop code (user code).
        # The issue is that the user code computes v_2 = ends - starts.
        # We can use that. But we need to know what variable name it has.

        # Actually, let's compute m_tiles from the local tile and sibling tile counts.
        # For a single sibling (N dim), the decomposition is:
        #   offset_jagged = (local_tile % m_tiles) * BS_jagged
        #   offset_sibling = (local_tile // m_tiles) * BS_sibling
        # We need m_tiles. We can get it from group_offsets:

        # Find the group_offsets device arg name
        group_offsets_dev_name = self._find_group_offsets_device_arg(device_function)

        offset_compute_stmts: list[ast.stmt] = [
            # Compute m_size from group_offsets
            statement_from_string(
                f"_m_size = tl.load({group_offsets_dev_name} + {group_idx_var} + 1) - "
                f"tl.load({group_offsets_dev_name} + {group_idx_var})"
            ),
            statement_from_string(
                f"{m_tiles_var} = ({env.backend.cdiv_expr('_m_size', jagged_bs_var, is_device=True)})"
            ),
        ]

        # Decompose local_tile into jagged offset and sibling offsets
        remaining = local_tile_var
        jagged_offset_stmt = statement_from_string(
            f"offset_{jagged_block_id} = ({remaining} % {m_tiles_var}) * {jagged_bs_var}"
        )
        offset_compute_stmts.append(jagged_offset_stmt)

        remaining_expr = f"({remaining} // {m_tiles_var})"
        for i, sib_id in enumerate(sibling_block_ids):
            sib_bs_var = device_function.block_size_var_cache.get(
                (sib_id,), f"_BLOCK_SIZE_{sib_id}"
            )
            if i + 1 < len(sibling_block_ids):
                # More siblings: use modulo
                next_sib_id = sibling_block_ids[i + 1]
                next_sib_info = env.block_sizes[next_sib_id]
                next_sib_numel = HostFunction.current().sympy_expr(next_sib_info.numel)
                next_sib_bs = device_function.block_size_var_cache.get(
                    (next_sib_id,), f"_BLOCK_SIZE_{next_sib_id}"
                )
                n_tiles = env.backend.cdiv_expr(
                    str(next_sib_numel), next_sib_bs, is_device=True
                )
                offset_compute_stmts.append(
                    statement_from_string(
                        f"offset_{sib_id} = ({remaining_expr} % ({n_tiles})) * {sib_bs_var}"
                    )
                )
                remaining_expr = f"({remaining_expr} // ({n_tiles}))"
            else:
                # Last sibling: no modulo needed
                offset_compute_stmts.append(
                    statement_from_string(
                        f"offset_{sib_id} = {remaining_expr} * {sib_bs_var}"
                    )
                )

        # 5. Assemble the flattened body
        # The new body inside the persistent loop is:
        #   decode_stmts (group_idx, local_tile)
        #   new_pre_loop (offset_0 = group_idx, indices_0, user code for starts/ends/M_g)
        #   offset_compute_stmts (m_tiles, offset_1 from local_tile, offset_2 from local_tile)
        #   indices/mask stmts from jagged loop setup (indices_1, mask_1)
        #   between_code (user code between jagged and sibling loops)
        #   indices/mask stmts from sibling loop setup (indices_2, etc.)
        #   innermost_body

        flat_body: list[ast.stmt] = []
        flat_body.extend(decode_stmts)
        flat_body.extend(new_pre_loop)
        flat_body.extend(offset_compute_stmts)

        # Now flatten the nested loop bodies (sibling for-loops are inlined,
        # reduction for-loops are preserved)
        self._flatten_loop_body(flat_body, jagged_body, sibling_block_ids)

        # 6. Wrap in persistent loop
        range_expr = (
            f"tl.range({typed_program_id(0)}, {total_tiles_var}, {self.grid_size_expr})"
        )

        persistent_loop = create(
            ast.For,
            target=create(ast.Name, id=virtual_pid_var, ctx=ast.Store()),
            iter=expr_from_string(range_expr),
            body=flat_body,
            orelse=[],
            type_comment=None,
        )

        return [persistent_loop]

    def _replace_grid_pid_with_group_idx(
        self, stmts: list[ast.stmt], group_idx_var: str
    ) -> list[ast.stmt]:
        """Replace pid_N = tl.program_id(0) and offset_N = pid_N with offset_N = group_idx."""
        result = []
        skip_pid_var: str | None = None
        for stmt in stmts:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = (
                    stmt.targets
                    if isinstance(stmt, ast.Assign)
                    else [stmt.target]
                    if stmt.target
                    else []
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Skip pid_N = tl.program_id(0) assignments
                        if name.startswith("pid_"):
                            skip_pid_var = name
                            break
                        # Replace offset_N = pid_N with offset_N = group_idx
                        if name.startswith("offset_") and skip_pid_var is not None:
                            result.append(
                                statement_from_string(f"{name} = {group_idx_var}")
                            )
                            skip_pid_var = None
                            break
                else:
                    result.append(stmt)
                    continue
                if skip_pid_var and name.startswith("pid_"):
                    continue  # Skip the pid assignment
            else:
                result.append(stmt)
        return result

    def _flatten_loop_body(
        self,
        flat_body: list[ast.stmt],
        loop_body: list[ast.stmt],
        remaining_sibling_ids: list[int],
    ) -> None:
        """Recursively flatten nested for-loops, keeping their setup stmts."""
        for stmt in loop_body:
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
                # Check if this is a sibling loop we want to flatten
                if target_name.startswith("offset_"):
                    try:
                        block_id = int(target_name.removeprefix("offset_"))
                    except ValueError:
                        flat_body.append(stmt)
                        continue
                    if block_id in remaining_sibling_ids:
                        # Flatten: add body contents instead of the for-loop
                        new_remaining = [
                            sid for sid in remaining_sibling_ids if sid != block_id
                        ]
                        self._flatten_loop_body(flat_body, stmt.body, new_remaining)
                        continue
                # Not a sibling loop (e.g. reduction K loop) — keep it
                flat_body.append(stmt)
            else:
                flat_body.append(stmt)

    def _find_for_node(self, stmts: list[ast.stmt], block_id: int) -> ast.For | None:
        """Find a for-loop node targeting offset_<block_id> in stmts (non-recursive)."""
        offset_name = f"offset_{block_id}"
        for stmt in stmts:
            if (
                isinstance(stmt, ast.For)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == offset_name
            ):
                return stmt
        return None

    def _collect_immediate_for_block_ids(self, stmts: list[ast.stmt]) -> list[int]:
        """Collect block_ids from for-loops that are immediate children of stmts.

        Only looks at the first for-loop found (not recursing into nested loops).
        """
        result = []
        for stmt in stmts:
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if name.startswith("offset_"):
                    try:
                        block_id = int(name.removeprefix("offset_"))
                    except ValueError:
                        continue
                    result.append(block_id)
        return result

    def _find_group_offsets_device_arg(self, device_function: DeviceFunction) -> str:
        """Find the device-side name of the group_offsets tensor argument."""
        for arg in device_function.arguments:
            if isinstance(arg, TensorArg) and not isinstance(arg, TensorDescriptorArg):
                if arg.fake_value.ndim == 1 and arg.fake_value.dtype in (
                    torch.int32,
                    torch.int64,
                ):
                    return arg.name
        raise ValueError(
            "Could not find group_offsets tensor argument for persistent_jagged"
        )
