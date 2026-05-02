from __future__ import annotations

import abc
import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.cutedsl_compat import emit_pipeline_advance
from .device_function import DeviceFunction
from .device_function import TensorArg
from .host_function import HostFunction


def typed_program_id(dim: int = 0) -> str:
    """Generate backend-specific program ID expression.

    Triton uses tl.program_id(). CuTe uses block_idx() as the virtual program ID.
    """
    env = CompileEnvironment.current()
    return env.backend.program_id_expr(dim, index_dtype=env.index_type())


def _stmt_name_uses(stmt: ast.AST) -> tuple[set[str], set[str]]:
    """Return ``(reads, writes)`` for the names referenced in ``stmt``."""
    reads: set[str] = set()
    writes: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                writes.add(node.id)
            else:
                reads.add(node.id)
    return reads, writes


if TYPE_CHECKING:
    import sympy

    from .device_function import CuteTcgen05MatmulPlan
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
        if not is_device:
            # Grid dimensions are always non-negative, so we can use integer
            # arithmetic directly instead of a function call like triton.cdiv.
            return f"(({numel_str}) + ({self.block_size_var}) - 1) // ({self.block_size_var})"
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


class Tcgen05PersistentProgramIDs(PersistentProgramIDs):
    """tcgen05 persistent scheduler for blocked and interleaved PID orders."""

    def __init__(self, *, is_blocked: bool) -> None:
        super().__init__(is_blocked=is_blocked)

    def _tcgen05_plan(self) -> CuteTcgen05MatmulPlan | None:
        return DeviceFunction.current().cute_tcgen05_matmul_plan

    def _tcgen05_cluster_m(self) -> int:
        if (plan := self._tcgen05_plan()) is not None:
            return plan.cluster_m
        config = DeviceFunction.current().config
        cluster_m = int(str(config.get("tcgen05_cluster_m", 1)))
        return max(1, min(cluster_m, 2))

    def _tcgen05_num_tiles_expr(self, *, is_device: bool) -> str:
        dims = [pid.num_pids_expr(is_device=is_device) for pid in self.pid_info]
        while len(dims) < 3:
            dims.append("1")
        return f"({', '.join(dims[:3])})"

    def _tcgen05_linear_virtual_pid_expr(self, work_tile_var: str) -> str:
        terms: list[str] = []
        for i, _pid in enumerate(self.pid_info):
            coord = f"{work_tile_var}.tile_idx[{i}]"
            if i == 0:
                terms.append(coord)
                continue
            stride = " * ".join(
                f"({pid.num_pids_expr(is_device=True)})" for pid in self.pid_info[:i]
            )
            terms.append(f"({coord}) * ({stride})")
        return " + ".join(terms) if terms else "cutlass.Int32(0)"

    def _tcgen05_linear_virtual_pid_from_coords_expr(self, coords: list[str]) -> str:
        terms: list[str] = []
        for i, coord in enumerate(coords[: len(self.pid_info)]):
            if i == 0:
                terms.append(coord)
                continue
            stride = " * ".join(
                f"({pid.num_pids_expr(is_device=True)})" for pid in self.pid_info[:i]
            )
            terms.append(f"({coord}) * ({stride})")
        return " + ".join(terms) if terms else "cutlass.Int32(0)"

    def _tcgen05_scheduler_owner_warp_expr(self) -> str:
        # ``Tcgen05PersistentProgramIDs`` is only instantiated when the kernel
        # selects tcgen05 MMA (see ``tile_strategy.select_pid_strategy``), and
        # ``cute_mma.py`` always registers the matmul plan in that path before
        # the persistent kernel setup runs.
        plan = self._tcgen05_plan()
        assert plan is not None, "tcgen05 persistent path requires a registered plan"
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.persistent_scheduler_owner_warp_id})"
        )

    def _tcgen05_scheduler_store_leader_expr(self) -> str:
        return (
            f"({self._tcgen05_scheduler_owner_warp_expr()}) "
            "and cute.arch.lane_idx() == cutlass.Int32(0)"
        )

    def _tcgen05_cluster_scheduler_leader_expr(self) -> str:
        if self._tcgen05_cluster_m() <= 1:
            return self._tcgen05_scheduler_store_leader_expr()
        return (
            f"({self._tcgen05_scheduler_owner_warp_expr()}) "
            "and cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == cutlass.Int32(0)"
        )

    def _tcgen05_store_work_tile_statements(
        self, work_tile_var: str, smem_var: str
    ) -> list[ast.stmt]:
        return [
            statement_from_string(
                f"{smem_var}[cutlass.Int32(0)] = {work_tile_var}.tile_idx[0]"
            ),
            statement_from_string(
                f"{smem_var}[cutlass.Int32(1)] = {work_tile_var}.tile_idx[1]"
            ),
            statement_from_string(
                f"{smem_var}[cutlass.Int32(2)] = {work_tile_var}.tile_idx[2]"
            ),
            statement_from_string(
                f"{smem_var}[cutlass.Int32(3)] = "
                f"(cutlass.Int32(1) if {work_tile_var}.is_valid_tile else cutlass.Int32(0))"
            ),
        ]

    def _tcgen05_scheduler_if(self, predicate: str, body: list[ast.stmt]) -> ast.If:
        return create(
            ast.If,
            test=expr_from_string(predicate),
            body=body,
            orelse=[],
        )

    def _split_tcgen05_invariant_setup(
        self, device_function: DeviceFunction, body: list[ast.stmt]
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """Split the device-function prefix into hoisted setup vs per-tile body.

        Codegen has explicitly tagged the per-tile statements via
        ``register_cute_tcgen05_per_tile_stmts``. Everything else can be
        hoisted out of the work-tile loop. This matches Quack's pattern of
        building pipelines once per kernel and replaying state per tile.

        The PID decomposition emitted by ``_decompose_virtual_pid``
        references ``virtual_pid_var`` (defined in the loop header) and
        produces ``pid_0``, ``pid_1`` etc. that are then consumed by
        downstream offset computations. To capture this transitive
        dependency without plumbing tagging through every codegen path, we
        do a single forward pass: seed the per-tile name set with
        ``virtual_pid_var``; any statement that reads or writes a per-tile
        name is itself per-tile, and any names it assigns become per-tile
        too.
        """
        if not device_function.has_cute_tcgen05_per_tile_marks:
            return [], body

        per_tile_names: set[str] = {self.virtual_pid_var}
        hoisted: list[ast.stmt] = []
        wrapped: list[ast.stmt] = []
        for stmt in body:
            reads, writes = _stmt_name_uses(stmt)
            is_per_tile = (
                device_function.is_cute_tcgen05_per_tile(stmt)
                or bool(reads & per_tile_names)
                or bool(writes & per_tile_names)
            )
            if is_per_tile:
                per_tile_names.update(writes)
                wrapped.append(stmt)
            else:
                hoisted.append(stmt)
        return hoisted, wrapped

    def _extract_tcgen05_post_loop_stmts(
        self, device_function: DeviceFunction, body: list[ast.stmt]
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """Pull post-loop tagged statements out of ``body``.

        Returns ``(remaining, post_loop)`` preserving relative order.

        Statements registered via ``register_cute_tcgen05_post_loop_stmts``
        belong after the persistent work-tile loop (one-shot drains:
        ``producer_tail``, TMEM dealloc, allocator setup). Without this
        extraction they would execute every tile, which wastes work and
        can corrupt pipeline state.
        """
        if not device_function.has_cute_tcgen05_post_loop_marks:
            return body, []
        remaining: list[ast.stmt] = []
        post_loop: list[ast.stmt] = []
        for stmt in body:
            if device_function.is_cute_tcgen05_post_loop(stmt):
                post_loop.append(stmt)
            else:
                remaining.append(stmt)
        return remaining, post_loop

    def _emit_host_multi_tile_guard(self, device_function: DeviceFunction) -> None:
        """Emit a host-side guard against multi-tile execution.

        Persistent + tcgen05 currently produces wrong output when the kernel
        processes more than one work tile in total. Empirically, even with
        148 SMs and 4 work tiles (so each CTA processes 0 or 1 tile), the
        persistent wrapper interferes with kernel correctness across tile
        boundaries. Only the single-tile case is verified correct. The fix
        is the role-local persistent rewrite; until that lands, this guard
        fails loudly when a user explicitly opts into a config whose problem
        shape exercises the broken path.

        The autotuner narrowing in ``matmul_ops.enforce_dot_requirements``
        already removes ``persistent_blocked`` / ``persistent_interleaved``
        from the search space for tcgen05 BF16/FP16 matmuls, so this guard
        only fires for explicit user configs that bypass autotune.

        The threshold is intentionally ``total_tiles > 1`` and not
        ``tiles_per_cta > 1`` or ``total_tiles > num_sms``: we have observed
        wrong output even when ``total_tiles <= num_sms`` (148 SMs, 4 work
        tiles). Loosening the guard to a per-CTA bound would re-introduce
        the silent wrong-output failure mode. Tighten only after the role-
        local persistent rewrite makes the multi-tile path actually
        correct.
        """
        host_total_pids = " * ".join(
            f"({pid.num_pids_expr(is_device=False)})" for pid in self.pid_info
        )
        if not host_total_pids:
            return
        # Bind the host-side total-tiles expression once so non-trivial pid-
        # count expressions are not duplicated in the emitted source. The
        # private name avoids colliding with user/host vars.
        total_var = "_helion_tcgen05_persistent_total_tiles"
        device_function.codegen.host_statements.append(
            statement_from_string(f"{total_var} = {host_total_pids}")
        )
        # Use %-style formatting in the generated code so the f-string braces
        # do not collide with ``statement_from_string`` placeholder syntax.
        guard = (
            f"if {total_var} > 1:\n"
            "    raise RuntimeError(\n"
            "        'Helion CuTe persistent + tcgen05 currently produces '\n"
            "        'wrong output when the kernel processes more than one '\n"
            "        'work tile total. The kernel was launched with '\n"
            "        'total_tiles=%d > 1, which exercises the multi-tile '\n"
            "        'path. Use a non-persistent pid_type (e.g. \"flat\") or '\n"
            "        'pick block sizes that keep total_tiles == 1.'"
            f"        % ({total_var},)\n"
            "    )"
        )
        device_function.codegen.host_statements.append(statement_from_string(guard))

    def _setup_tcgen05_persistent_kernel(
        self, device_function: DeviceFunction
    ) -> list[ast.stmt]:
        self._emit_host_multi_tile_guard(device_function)
        wrapped_body = cast("list[ast.stmt]", list(device_function.body))
        if isinstance(device_function.pid, ForEachProgramID):
            shared_pid_var = device_function.pid.shared_pid_var
            wrapped_body = [
                statement_from_string(f"{shared_pid_var} = {self.virtual_pid_var}"),
                *wrapped_body,
            ]
        # Order matters: pull post-loop cleanup out FIRST so the per-tile
        # splitter never has a chance to trace those statements into the
        # work-tile body via name propagation. Reversing this would re-
        # introduce the dominance-error class of bugs that motivated the
        # post-loop tag.
        wrapped_body, post_loop_stmts = self._extract_tcgen05_post_loop_stmts(
            device_function, wrapped_body
        )
        hoisted_setup, wrapped_body = self._split_tcgen05_invariant_setup(
            device_function, wrapped_body
        )

        cluster_m = self._tcgen05_cluster_m()
        tile_sched_params_var = device_function.new_var("tcgen05_tile_sched_params")
        tile_sched_var = device_function.new_var("tcgen05_tile_sched")
        work_tile_var = device_function.new_var("tcgen05_work_tile")
        work_tile_smem_ptr = device_function.new_var("tcgen05_work_tile_smem_ptr")
        work_tile_smem = device_function.new_var("tcgen05_work_tile_smem")
        work_tile_smem_tensor = device_function.new_var("tcgen05_work_tile_smem_tensor")
        work_tile_coord_vars = [
            device_function.new_var(f"tcgen05_work_tile_idx_{i}") for i in range(3)
        ]
        work_tile_valid_var = device_function.new_var("tcgen05_work_tile_valid")
        scheduler_owner_warp = self._tcgen05_scheduler_owner_warp_expr()
        cluster_scheduler_leader = self._tcgen05_cluster_scheduler_leader_expr()
        consumer_leader = device_function.new_var("tcgen05_sched_consumer_leader")
        linear_pid_expr = self._tcgen05_linear_virtual_pid_from_coords_expr(
            work_tile_coord_vars
        )
        sched_pipeline_mbars = device_function.new_var("tcgen05_sched_pipeline_mbars")
        sched_pipeline = device_function.new_var("tcgen05_sched_pipeline")
        sched_pipeline_producer_group = device_function.new_var(
            "tcgen05_sched_pipeline_producer_group"
        )
        sched_pipeline_consumer_group = device_function.new_var(
            "tcgen05_sched_pipeline_consumer_group"
        )
        sched_producer_state = device_function.new_var("tcgen05_sched_producer_state")
        sched_consumer_state = device_function.new_var("tcgen05_sched_consumer_state")
        sched_barrier_ptr = device_function.new_var("tcgen05_sched_barrier_ptr")
        sched_peer_rank = device_function.new_var("tcgen05_sched_peer_rank")
        sched_peer_m = device_function.new_var("tcgen05_sched_peer_m")

        refresh_work_tile = [
            statement_from_string(f"{coord_var} = {work_tile_smem}[cutlass.Int32({i})]")
            for i, coord_var in enumerate(work_tile_coord_vars)
        ]
        refresh_work_tile.append(
            statement_from_string(
                f"{work_tile_valid_var} = "
                f"{work_tile_smem}[cutlass.Int32(3)] != cutlass.Int32(0)"
            )
        )
        if cluster_m > 1:
            work_tile_publish = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                statement_from_string(
                    f"{sched_barrier_ptr} = {sched_pipeline}.producer_get_barrier({sched_producer_state})"
                ),
                statement_from_string(f"{sched_peer_rank} = cute.arch.lane_idx()"),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{sched_peer_rank} < cutlass.Int32({cluster_m})"
                    ),
                    body=[
                        statement_from_string(
                            f"{sched_peer_m} = {sched_peer_rank} % cutlass.Int32({cluster_m})"
                        ),
                        statement_from_string(
                            f"_cute_store_shared_remote_x4("
                            f"{work_tile_var}.tile_idx[0] + {sched_peer_m}, "
                            f"{work_tile_var}.tile_idx[1], "
                            f"{work_tile_var}.tile_idx[2], "
                            f"(cutlass.Int32(1) if {work_tile_var}.is_valid_tile else cutlass.Int32(0)), "
                            f"smem_ptr={work_tile_smem_ptr}, "
                            f"mbar_ptr={sched_barrier_ptr}, "
                            f"peer_cta_rank_in_cluster={sched_peer_rank})"
                        ),
                    ],
                    orelse=[],
                ),
                statement_from_string(
                    f"{sched_pipeline}.producer_commit({sched_producer_state})"
                ),
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
            ]
            work_tile_consume = [
                statement_from_string(
                    f"{sched_pipeline}.consumer_wait({sched_consumer_state})"
                ),
                statement_from_string("cute.arch.fence_view_async_shared()"),
                statement_from_string("cute.arch.sync_warp()"),
            ]
            work_tile_release = [
                statement_from_string(
                    f"{sched_pipeline}.consumer_release({sched_consumer_state})"
                ),
                statement_from_string(emit_pipeline_advance(sched_consumer_state)),
            ]
        else:
            work_tile_publish = self._tcgen05_store_work_tile_statements(
                work_tile_var, work_tile_smem
            )
            work_tile_consume = []
            work_tile_release = []

        loop_body = [
            statement_from_string(f"{self.virtual_pid_var} = {linear_pid_expr}"),
            *wrapped_body,
            self._tcgen05_scheduler_if(
                cluster_scheduler_leader if cluster_m > 1 else scheduler_owner_warp,
                [
                    statement_from_string(f"{tile_sched_var}.advance_to_next_work()"),
                    statement_from_string(
                        f"{work_tile_var} = {tile_sched_var}.get_current_work()"
                    ),
                    *work_tile_publish,
                ],
            ),
            *(
                [
                    self._tcgen05_scheduler_if(consumer_leader, work_tile_consume),
                ]
                if cluster_m > 1
                else []
            ),
            statement_from_string("cute.arch.sync_threads()"),
            *refresh_work_tile,
            *(
                [
                    self._tcgen05_scheduler_if(consumer_leader, work_tile_release),
                ]
                if cluster_m > 1
                else []
            ),
        ]
        setup = [
            statement_from_string(
                f"{tile_sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                f"{self._tcgen05_num_tiles_expr(is_device=True)}, "
                f"({cluster_m}, 1, 1))"
            ),
            statement_from_string(
                f"{tile_sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                f"{tile_sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
            ),
            statement_from_string(
                f"{work_tile_smem_ptr} = cute.arch.alloc_smem(cutlass.Int32, 4, alignment=16)"
            ),
            statement_from_string(
                f"{work_tile_smem_tensor} = cute.make_tensor("
                f"{work_tile_smem_ptr}, cute.make_layout((4,), stride=(1,)))"
            ),
            statement_from_string(f"{work_tile_smem} = {work_tile_smem_tensor}"),
        ]
        if cluster_m > 1:
            setup.extend(
                [
                    statement_from_string(
                        f"{sched_pipeline_mbars} = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(2))"
                    ),
                    statement_from_string(
                        f"{sched_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
                        "cutlass.pipeline.Agent.Thread, cute.arch.WARP_SIZE)"
                    ),
                    statement_from_string(
                        f"{sched_pipeline_consumer_group} = cutlass.pipeline.CooperativeGroup("
                        f"cutlass.pipeline.Agent.Thread, {cluster_m})"
                    ),
                    statement_from_string(
                        f"{sched_pipeline} = cutlass.pipeline.PipelineAsync.create("
                        "num_stages=1, "
                        f"producer_group={sched_pipeline_producer_group}, "
                        f"consumer_group={sched_pipeline_consumer_group}, "
                        f"barrier_storage={sched_pipeline_mbars}, "
                        "consumer_mask=cutlass.Int32(0), "
                        "defer_sync=True)"
                    ),
                    statement_from_string(
                        f"{sched_producer_state} = cutlass.pipeline.make_pipeline_state("
                        "cutlass.pipeline.PipelineUserType.Producer, 1)"
                    ),
                    statement_from_string(
                        f"{sched_consumer_state} = cutlass.pipeline.make_pipeline_state("
                        "cutlass.pipeline.PipelineUserType.Consumer, 1)"
                    ),
                    statement_from_string(
                        f"{consumer_leader} = cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(0) "
                        "and cute.arch.lane_idx() == cutlass.Int32(0)"
                    ),
                ]
            )
        else:
            setup.append(statement_from_string(f"{consumer_leader} = False"))
        setup.extend(
            [
                self._tcgen05_scheduler_if(
                    cluster_scheduler_leader if cluster_m > 1 else scheduler_owner_warp,
                    [
                        statement_from_string(
                            f"{work_tile_var} = {tile_sched_var}.initial_work_tile_info()"
                        ),
                        *work_tile_publish,
                    ],
                ),
            ]
        )
        if cluster_m > 1:
            setup.append(self._tcgen05_scheduler_if(consumer_leader, work_tile_consume))
        setup.extend(
            [
                statement_from_string("cute.arch.sync_threads()"),
                *refresh_work_tile,
            ]
        )
        if cluster_m > 1:
            setup.append(self._tcgen05_scheduler_if(consumer_leader, work_tile_release))
        setup.extend(
            [
                *hoisted_setup,
                create(
                    ast.While,
                    test=expr_from_string(work_tile_valid_var),
                    body=loop_body,
                    orelse=[],
                ),
                *post_loop_stmts,
            ]
        )
        return setup

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        return self._setup_tcgen05_persistent_kernel(device_function)
