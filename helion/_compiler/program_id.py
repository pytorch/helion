from __future__ import annotations

import abc
import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import ClassVar
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
from .host_function import NoCurrentFunction


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
        try:
            return DeviceFunction.current().cute_tcgen05_matmul_plan
        except NoCurrentFunction:
            # Unit tests exercise builder helpers without entering a
            # DeviceFunction; in that context the tcgen05 plan-dependent
            # branches should behave like the legacy 1-CTA path.
            return None

    def _tcgen05_cluster_m(self) -> int:
        if (plan := self._tcgen05_plan()) is not None:
            return plan.cluster_m
        config = DeviceFunction.current().config
        cluster_m = int(str(config.get("tcgen05_cluster_m", 1)))
        return max(1, min(cluster_m, 2))

    def _tcgen05_is_two_cta(self) -> bool:
        if (plan := self._tcgen05_plan()) is not None:
            return plan.is_two_cta
        return False

    def _tcgen05_output_tile_dims_expr(self, *, is_device: bool) -> list[str]:
        assert len(self.pid_info) <= 3, (
            "tcgen05 persistent scheduler supports at most 3 PID dimensions"
        )
        dims = [pid.num_pids_expr(is_device=is_device) for pid in self.pid_info]
        while len(dims) < 3:
            dims.append("1")
        return dims

    def _tcgen05_scheduler_tile_dims_expr(self, *, is_device: bool) -> list[str]:
        dims = self._tcgen05_output_tile_dims_expr(is_device=is_device)
        if self._tcgen05_is_two_cta():
            # CtaGroup.TWO uses two CTAs to produce one logical M tile. Model
            # scheduler M as CTA slots, then collapse back to logical M when
            # binding virtual_pid for PID decomposition.
            dims[0] = f"({dims[0]}) * {self._tcgen05_cluster_m()}"
        return dims

    def _tcgen05_num_tiles_expr(self, *, is_device: bool) -> str:
        dims = self._tcgen05_scheduler_tile_dims_expr(is_device=is_device)
        return f"({', '.join(dims[:3])})"

    def _tcgen05_num_work_clusters_expr(self, *, is_device: bool) -> str:
        """Return the number of scheduler work clusters.

        ``StaticPersistentTileScheduler.create`` initializes its current
        work index from ``block_idx.z`` and uses ``block_idx.x/y`` only as
        the CTA's coordinate inside a cluster. The launch grid therefore
        needs one z block per persistent work cluster, not a flat x-only
        ``(_NUM_SM,)`` grid.
        """
        dims = self._tcgen05_scheduler_tile_dims_expr(is_device=is_device)
        cluster_m = self._tcgen05_cluster_m()
        if cluster_m > 1:
            dims[0] = f"(({dims[0]}) + {cluster_m} - 1) // {cluster_m}"
        return " * ".join(f"({dim})" for dim in dims[:3])

    def codegen_grid(self) -> ast.AST:
        # Tcgen05 persistent kernels, including guarded legacy fallback
        # shapes, use CUTLASS' z-indexed scheduler instead of the parent
        # virtual-PID loop. Multi-root ForEach kernels are still host-guarded
        # because this grid is derived from this case's pid_info only.
        cluster_m = self._tcgen05_cluster_m()
        total_clusters = self._tcgen05_num_work_clusters_expr(is_device=False)
        if cluster_m == 1:
            max_persistent_clusters = self.grid_size_expr
        else:
            # ``grid_size_expr`` is expressed in CTAs. A tcgen05 scheduler
            # work cluster owns ``cluster_m`` CTAs along M, so z may contain
            # at most the CTA budget divided by that cluster size. Explicit
            # cluster_m > 1 configs still codegen, but multi-tile runtime is
            # guarded until that lowering is validated.
            max_persistent_clusters = f"max(1, ({self.grid_size_expr}) // {cluster_m})"
        return expr_from_string(
            f"({cluster_m}, 1, min(({total_clusters}), ({max_persistent_clusters})))"
        )

    def _tcgen05_logical_m_coord_expr(self, coord: str) -> str:
        if self._tcgen05_is_two_cta():
            return f"({coord}) // cutlass.Int32({self._tcgen05_cluster_m()})"
        return coord

    def _tcgen05_linear_virtual_pid_expr(self, work_tile_var: str) -> str:
        terms: list[str] = []
        for i, _pid in enumerate(self.pid_info):
            coord = f"{work_tile_var}.tile_idx[{i}]"
            if i == 0:
                terms.append(self._tcgen05_logical_m_coord_expr(coord))
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
                terms.append(self._tcgen05_logical_m_coord_expr(coord))
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

    def _tcgen05_exec_warp_expr(self) -> str:
        plan = self._tcgen05_plan()
        assert plan is not None, "tcgen05 persistent path requires a registered plan"
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.exec_warp_id})"
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

    def _retarget_tcgen05_shared_scheduler_to_exec(
        self, layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout
    ) -> None:
        """Make the shared persistent loop's scheduler live on the exec warp.

        Once the TMA-load warp is lifted into a role-local sibling loop, the
        scheduler should not ride on that producer role. The exec warp remains
        a single, always-launched warp that rejoins the shared loop after any
        role-local work, making it a stable owner for the intermediate G1
        shapes until a dedicated scheduler or fully role-local epilogue lands.
        """
        exec_warp = self._tcgen05_exec_warp_expr()
        layout.scheduler_owner_warp = exec_warp
        layout.cluster_scheduler_leader = (
            f"({exec_warp}) "
            "and cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == cutlass.Int32(0)"
        )
        layout.scheduler_leader_predicate = (
            layout.cluster_scheduler_leader if layout.cluster_m > 1 else exec_warp
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

    def _tcgen05_tma_load_role_predicate(self) -> str:
        """Boolean expression that gates the TMA-load warp's role block.

        ``CuteTcgen05MatmulPlan.tma_warp_id`` is the launched-CTA warp
        index assigned to TMA load + (currently) the persistent
        scheduler. Match the tagging that ``cute_mma.py`` already emits
        (``f"{tma_warp} = {warp_idx} == cutlass.Int32({tma_warp_id})"``)
        so the predicate evaluates the same on every warp.
        """
        plan = self._tcgen05_plan()
        assert plan is not None, (
            "tcgen05 TMA-load role predicate requires a registered matmul plan"
        )
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.tma_warp_id})"
        )

    def _tcgen05_mma_exec_role_predicate(self) -> str:
        """Boolean expression that gates the MMA-exec warp's role block."""
        plan = self._tcgen05_plan()
        assert plan is not None, (
            "tcgen05 MMA-exec role predicate requires a registered matmul plan"
        )
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.exec_warp_id})"
        )

    def _tcgen05_epi_role_predicate(self) -> str:
        """Boolean expression that gates the epilogue warps' role block."""
        plan = self._tcgen05_plan()
        assert plan is not None, (
            "tcgen05 epilogue role predicate requires a registered matmul plan"
        )
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"< cutlass.Int32({plan.epi_warp_count})"
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

    def _collect_tcgen05_role_blocks(
        self, device_function: DeviceFunction, body: list[ast.stmt]
    ) -> list[Tcgen05PersistentProgramIDs._PersistentRoleBlock]:
        """Partition the per-tile body into warp-role blocks (inline weave).

        Thin wrapper around :meth:`_partition_tcgen05_role_blocks` that
        flattens the partitioned result back into a single linear sequence
        of role blocks preserving the original body's emit order. This is
        the legacy producer used by
        :meth:`_build_tcgen05_persistent_tile_body` for the single-shared-
        ``while`` path -- TMA-load role blocks become inline
        ``if {tma_warp_predicate}: ...`` wrappers in their original
        positions inside the per-tile body.

        See :meth:`_partition_tcgen05_role_blocks` for the lower-level
        contract that the role-local-while consumer in
        :meth:`_build_tcgen05_persistent_tile_body_role_local` consumes
        directly.
        """
        partition = self._partition_tcgen05_role_blocks(device_function, body)
        return partition.role_blocks_inline

    def _partition_tcgen05_role_blocks(
        self, device_function: DeviceFunction, body: list[ast.stmt]
    ) -> Tcgen05PersistentProgramIDs._PartitionedRoleBody:
        """Walk the per-tile body and produce a structured role-block partition.

        Returns a :class:`_PartitionedRoleBody` carrying:

        - ``role_blocks_inline``: the legacy linear sequence of role
          blocks preserving the original emit order. Top-level
          role-tagged statements stay sandwiched between shared blocks
          here, ready for the inline-weave consumer to wrap them in
          ``if {role_predicate}: ...``.
        - ``role_blocks_extracted``: each non-shared role block as a
          standalone unit, decoupled from any surrounding shared
          statements. The extract-and-remove consumer in
          :meth:`_build_tcgen05_persistent_tile_body_role_local` lifts
          these into role-local ``while`` loops.
        - ``shared_body_extracted``: the original ``body`` with every
          top-level tagged statement removed. The extract consumer
          weaves this into the shared ``while`` while the extracted
          role blocks fill the role-local ``while`` siblings.

        The producer walks the body in order. Each maximal run of
        consecutive statements tagged for the same role is collapsed into
        a role block gated by that role's warp predicate. Everything else
        lives in the surrounding shared blocks. This preserves the
        original emit order for the inline view: role-tagged statements
        sandwiched between shared statements stay sandwiched, only wrapped
        in a role-gate ``if``. The extracted view removes top-level role
        blocks from the shared body and emits them as role-local sibling
        ``while`` loops.

        Some tagged statements still gate themselves on role predicates
        inline. In the inline view that is functionally redundant with the
        outer role-block ``if``. In the extracted role-local path, newly
        split producer / exec K-loops can drop those inline gates because
        the enclosing role-local ``while`` already restricts execution.

        When no role tags are present, the producer returns a single
        shared block carrying the full body. This is the non-tcgen05 path,
        the universal-MMA path, and any kernel that never registers role
        tags. The consumer
        (``_build_tcgen05_persistent_tile_body``) handles the
        single-block case identically to the pre-split implementation.

        **Nested tags inside top-level loops.** The K-loop's per-iter
        role blocks can be emitted INSIDE the K-loop body via
        ``cg.add_statement(...)``, so they are not top-level statements
        of the per-tile body. Tagged statements found inside top-level
        ``for`` / ``while`` loop bodies get rewritten in place: each
        tagged child statement is wrapped with
        ``if {role_predicate}: <child>`` so the role gate is visible in
        the generated source. This legacy inline path is still used for
        shapes that do not enter the role-local static-full path.

        Recursion is intentionally one level deep: the K-loop is the
        only top-level loop the role partitioner needs to reach into
        today, and a one-level recursion keeps the code simple. If
        future codegen places tagged statements inside nested loops the
        recursion can be deepened then.
        """
        tma_load_predicate = self._tcgen05_tma_load_role_predicate()
        mma_exec_predicate = self._tcgen05_mma_exec_role_predicate()
        epi_predicate = self._tcgen05_epi_role_predicate()
        role_predicates_by_id: dict[int, str] = {}
        for stmt_id in device_function.cute_tcgen05_tma_load_role_stmt_ids:
            role_predicates_by_id[stmt_id] = tma_load_predicate
        for stmt_id in device_function.cute_tcgen05_mma_exec_role_stmt_ids:
            assert stmt_id not in role_predicates_by_id, (
                "tcgen05 role statement registered for multiple warp roles"
            )
            role_predicates_by_id[stmt_id] = mma_exec_predicate
        for stmt_id in device_function.cute_tcgen05_epi_role_stmt_ids:
            assert stmt_id not in role_predicates_by_id, (
                "tcgen05 role statement registered for multiple warp roles"
            )
            role_predicates_by_id[stmt_id] = epi_predicate

        if not role_predicates_by_id:
            single = self._PersistentRoleBlock(role_predicate=None, stmts=list(body))
            return self._PartitionedRoleBody(
                role_blocks_inline=[single],
                role_blocks_extracted=[],
                shared_body_extracted=list(body),
            )

        inline_blocks: list[Tcgen05PersistentProgramIDs._PersistentRoleBlock] = []
        extracted_blocks: list[Tcgen05PersistentProgramIDs._PersistentRoleBlock] = []
        shared_body_extracted: list[ast.stmt] = []
        current_shared: list[ast.stmt] = []
        current_role_predicate: str | None = None
        current_role_stmts: list[ast.stmt] = []
        # Track every role-tag id the partitioner consumes so we can
        # detect a registered tag that never landed in a role block --
        # i.e. a top-level tag that was hoisted out of the work-tile
        # body before the partitioner ran, or a tag buried in a
        # container the recursion does not enter (anything other than
        # a top-level ``for`` / ``while``). Either case would silently
        # drop the role gate, so we assert below.
        visited_role_ids: set[int] = set()

        def role_predicate_for(stmt: ast.stmt) -> str | None:
            return role_predicates_by_id.get(id(stmt))

        def flush_shared() -> None:
            if current_shared:
                inline_blocks.append(
                    self._PersistentRoleBlock(
                        role_predicate=None, stmts=list(current_shared)
                    )
                )
                current_shared.clear()

        def flush_role() -> None:
            nonlocal current_role_predicate
            if current_role_stmts:
                assert current_role_predicate is not None
                # The inline view holds the role block in its original
                # position so the inline-weave consumer keeps the
                # defines-before-uses invariant unchanged. The extracted
                # view holds a structurally-separated copy of the same
                # statements so the role-local-while consumer can lift
                # them into a sibling ``while`` without disturbing the
                # shared body's order.
                inline_blocks.append(
                    self._PersistentRoleBlock(
                        role_predicate=current_role_predicate,
                        stmts=list(current_role_stmts),
                    )
                )
                extracted_blocks.append(
                    self._PersistentRoleBlock(
                        role_predicate=current_role_predicate,
                        stmts=list(current_role_stmts),
                    )
                )
                current_role_stmts.clear()
                current_role_predicate = None

        def wrap_nested_role_in_for_or_while(stmt: ast.stmt) -> None:
            """Walk a top-level ``for`` / ``while`` body; wrap tagged
            children in ``if {role_predicate}: <child>``. Mutates the
            loop body in place so the loop emits with role gating in
            place of the original child."""
            if not isinstance(stmt, (ast.For, ast.While)):
                return
            new_body: list[ast.stmt] = []
            for child in stmt.body:
                child_predicate = role_predicate_for(child)
                if child_predicate is not None:
                    visited_role_ids.add(id(child))
                    new_body.append(
                        create(
                            ast.If,
                            test=expr_from_string(child_predicate),
                            body=[child],
                            orelse=[],
                        )
                    )
                else:
                    new_body.append(child)
            stmt.body = new_body

        for stmt in body:
            role_predicate = role_predicate_for(stmt)
            if role_predicate is not None:
                flush_shared()
                visited_role_ids.add(id(stmt))
                if (
                    current_role_predicate is not None
                    and current_role_predicate != role_predicate
                ):
                    flush_role()
                current_role_predicate = role_predicate
                current_role_stmts.append(stmt)
            else:
                flush_role()
                wrap_nested_role_in_for_or_while(stmt)
                current_shared.append(stmt)
                shared_body_extracted.append(stmt)
        flush_shared()
        flush_role()

        registered_role_ids = frozenset(role_predicates_by_id)
        missed_ids = registered_role_ids - visited_role_ids
        assert not missed_ids, (
            f"{len(missed_ids)} tcgen05 role-tagged statement(s) were "
            "registered but not visited by the role partitioner. Top-level "
            "tagged stmts must also be per-tile-registered (otherwise the "
            "splitter hoists them out of the work-tile body before the "
            "partitioner runs); nested tagged stmts must be direct children "
            "of a top-level ``for`` / ``while`` in the per-tile body (the "
            "recursion is one level deep and does not enter ``if`` / other "
            "containers)."
        )
        return self._PartitionedRoleBody(
            role_blocks_inline=inline_blocks,
            role_blocks_extracted=extracted_blocks,
            shared_body_extracted=shared_body_extracted,
        )

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

    # Host-side variable that binds the total-tile expression once so the
    # guard message can format it. Private name avoids user/host collisions.
    _MULTI_TILE_GUARD_TOTAL_VAR: ClassVar[str] = (
        "_helion_tcgen05_persistent_total_tiles"
    )

    # Error message body for the multi-tile guard. Kept as a class constant so
    # the test pin and the error path stay in sync. ``%d`` is filled in at
    # runtime with the bound total-tile count.
    _MULTI_TILE_GUARD_MESSAGE: ClassVar[str] = (
        "Helion CuTe persistent + tcgen05 currently supports multi-tile "
        "execution only for single-root static full tiles with "
        "tcgen05_cluster_m=1. Partial K/M/N tile fallback shapes, "
        "multi-root kernels, and unvalidated cluster_m settings can "
        "produce wrong output, hang, or launch-fail. The kernel was launched "
        "with total_tiles=%d, which is outside the validated persistent "
        "scheduler set for this path. "
        'Use a non-persistent pid_type (e.g. "flat"), pick a single-root '
        "static-full-tile kernel with tcgen05_cluster_m=1, or, for "
        "single-root fallback paths, pick block sizes that keep "
        "total_tiles == 1."
    )

    def _emit_host_multi_tile_guard(
        self,
        device_function: DeviceFunction,
        host_total_pids_expr: str | None = None,
        guard_threshold: int = 1,
    ) -> None:
        """Emit a host-side guard against multi-tile execution.

        The single-root static full-tile role-local path has multi-tile
        runtime coverage only for ``tcgen05_cluster_m == 1``. Legacy
        non-role-local tcgen05 persistent kernels, multi-root kernels, and
        cluster_m > 1 configs still hit, or lack coverage for, wrong-output /
        hang / launch-failure modes, so this guard remains for those paths.
        Single-tile fallback shapes with ``cluster_m == 1`` continue to run
        because they do not exercise scheduler tile-to-tile transitions.

        The autotuner narrowing in
        ``ConfigSpec.narrow_tcgen05_autotune_to_validated_configs`` removes
        ``persistent_blocked`` / ``persistent_interleaved`` from the search
        space for tcgen05 BF16/FP16 matmuls, so this guard only fires for
        explicit user configs that bypass autotune.

        The threshold is intentionally ``total_tiles > 1`` for guarded
        single-root paths. For partial fallback shapes this converts known
        launch and wrong-output failures into a host error. Multi-root
        kernels use ``total_tiles > 0`` because the scheduler grid is
        derived from only the first root case; even one tile in a later case
        is unsafe. Cluster_m > 1 configs also use ``total_tiles > 0`` because
        CtaGroup.TWO codegen is structural-only until G3 runtime ownership is
        validated.
        """
        host_total_pids = host_total_pids_expr
        if host_total_pids is None:
            host_total_pids = " * ".join(
                f"({pid.num_pids_expr(is_device=False)})" for pid in self.pid_info
            )
        if not host_total_pids:
            return
        # Bind the host-side total-tiles expression once so non-trivial pid-
        # count expressions are not duplicated in the emitted source.
        total_var = self._MULTI_TILE_GUARD_TOTAL_VAR
        device_function.codegen.host_statements.append(
            statement_from_string(f"{total_var} = {host_total_pids}")
        )
        # Use ``repr()`` so the literal survives ``statement_from_string``
        # placeholder parsing (``{word}`` is reserved); ``%d`` interpolates
        # the total-tile count at runtime.
        message_literal = repr(self._MULTI_TILE_GUARD_MESSAGE)
        guard = (
            f"if {total_var} > {guard_threshold}:\n"
            f"    raise RuntimeError({message_literal} % ({total_var},))"
        )
        device_function.codegen.host_statements.append(statement_from_string(guard))

    def _setup_tcgen05_persistent_kernel(
        self,
        device_function: DeviceFunction,
    ) -> list[ast.stmt]:
        wrapped_body = cast("list[ast.stmt]", list(device_function.body))
        multi_root_pid = device_function.pid
        is_multi_root = isinstance(multi_root_pid, ForEachProgramID)
        host_guard_total_pids = None
        if is_multi_root:
            assert isinstance(multi_root_pid, ForEachProgramID)
            shared_pid_var = multi_root_pid.shared_pid_var
            host_guard_total_pids = multi_root_pid.total_pids_expr(is_device=False)
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

        layout = self._build_tcgen05_persistent_layout(device_function)
        partition = self._partition_tcgen05_role_blocks(device_function, wrapped_body)
        use_role_local_body = bool(partition.role_blocks_extracted)
        use_validated_role_local_body = (
            use_role_local_body and layout.cluster_m == 1 and not is_multi_root
        )
        if use_role_local_body:
            # Retarget even for guarded cluster_m>1 / multi-root codegen so
            # compile-only inspection still sees the role-local scheduler shape.
            self._retarget_tcgen05_shared_scheduler_to_exec(layout)
        if not use_validated_role_local_body:
            self._emit_host_multi_tile_guard(
                device_function,
                host_guard_total_pids,
                guard_threshold=0 if is_multi_root or layout.cluster_m > 1 else 1,
            )

        setup: list[ast.stmt] = []
        setup.extend(self._build_tcgen05_persistent_prelude(layout))
        setup.extend(hoisted_setup)
        if use_role_local_body:
            role_local_whiles, shared_tile_body = (
                self._build_tcgen05_persistent_tile_body_role_local(
                    device_function, layout, partition
                )
            )
            setup.extend(role_local_whiles)
            # All warps still enter the shared loop after any role-local
            # mainloop work so existing CTA-wide barriers remain valid for
            # epilogue synchronization and work-tile metadata publication.
            setup.append(
                create(
                    ast.While,
                    test=expr_from_string(layout.work_tile_valid_var),
                    body=shared_tile_body,
                    orelse=[],
                )
            )
        else:
            setup.append(
                create(
                    ast.While,
                    test=expr_from_string(layout.work_tile_valid_var),
                    body=self._build_tcgen05_persistent_tile_body(
                        layout, partition.role_blocks_inline
                    ),
                    orelse=[],
                )
            )
        setup.extend(post_loop_stmts)
        return setup

    @dataclasses.dataclass
    class _PersistentRoleBlock:
        """One warp-role's contribution to the per-tile work-tile body.

        Each role block carries the statements that conceptually belong
        to one warp role (TMA-load / MMA-exec / epi / scheduler), plus a
        ``role_predicate`` boolean expression that evaluates true on the
        warps that should run those statements. ``role_predicate is
        None`` denotes a "shared" block that runs on every warp -- this
        is the default for kernel statements that have no explicit role
        tag (e.g. PID decomposition, offset compute, cross-role
        ``cute.arch.sync_threads()`` calls).

        The legacy consumer
        (:meth:`_build_tcgen05_persistent_tile_body`) emits each role
        block sequentially inside the single shared work-tile ``while``:
        shared blocks become naked statements, role-gated blocks become
        ``if {role_predicate}: ...`` wrappers. This is functionally
        equivalent to the pre-split persistent body because every
        role-tagged statement was already gated on the same predicate
        inside its emit site (e.g. the initial TMA prefetch was already
        wrapped in ``if {tma_warp}:`` in ``cute_mma.py``).

        The role-local consumer
        (:meth:`_build_tcgen05_persistent_tile_body_role_local`) emits
        one role-local ``while`` per unique role predicate driven by
        its own scheduler instance. In the current mainloop-role
        intermediate, every
        warp still enters the shared body after any role-local work so
        existing CTA-wide ``cute.arch.sync_threads()`` barriers remain
        valid. The AB / acc pipelines carry producer-consumer ordering
        between the role-local TMA producer, role-local MMA exec, and
        shared epilogue consumer.
        """

        role_predicate: str | None
        stmts: list[ast.stmt]

    @dataclasses.dataclass
    class _PartitionedRoleBody:
        """Structured result of :meth:`_partition_tcgen05_role_blocks`.

        Carries three views of the same per-tile body so the inline-
        weave consumer and the role-local-while consumer can each pick
        the form that matches their emission shape:

        - ``role_blocks_inline``: the legacy linear sequence of role
          blocks preserving the original emit order. Top-level
          TMA-load-tagged statements appear sandwiched between shared
          blocks here, ready for the inline-weave consumer to wrap them
          in ``if {role_predicate}: ...``. When tagged statements are
          nested inside a top-level ``for`` / ``while``, the partitioner
          mutates the loop body in place to wrap the tagged child in an
          ``if {role_predicate}:``; the (now-mutated) loop appears in
          this view.
        - ``role_blocks_extracted``: each non-shared run of TMA-load-
          tagged top-level statements as a standalone block, decoupled
          from any surrounding shared statements. The role-local-while
          consumer lifts these into role-local ``while`` siblings.
          Nested tagged statements (inside top-level ``for`` / ``while``
          bodies) are NOT extracted; they stay inside their containing
          loop in ``role_blocks_inline`` / ``shared_body_extracted``.
        - ``shared_body_extracted``: the original ``body`` with every
          top-level tagged statement removed. Note that any top-level
          ``for`` / ``while`` containing nested tagged children appears
          here in its mutated (inline-wrapped) form, so this view is
          fully decoupled from ``role_blocks_extracted`` only when the
          partitioner did not need to recurse.

        The top-level lists are independent: mutating the elements list
        of one view does not affect another. The contained ``ast.stmt``
        nodes, however, are shared across views by reference -- mutating
        a node in place (e.g. wrapping it in an ``ast.If``) is visible
        from every view that references it. Consumers that need to
        rewrite an AST node should ``ast.copy_location`` / construct a
        fresh node rather than mutate in place.
        """

        role_blocks_inline: list[Tcgen05PersistentProgramIDs._PersistentRoleBlock]
        role_blocks_extracted: list[Tcgen05PersistentProgramIDs._PersistentRoleBlock]
        shared_body_extracted: list[ast.stmt]

    @dataclasses.dataclass
    class _Tcgen05PersistentLayout:
        """Variables and predicates threaded through the persistent kernel.

        The layout is materialised once per kernel and shared between the
        prelude (pre-loop init) and the per-tile body. Cluster-only
        fields are unused when ``cluster_m == 1``.
        """

        cluster_m: int
        scheduler_owner_warp: str
        cluster_scheduler_leader: str
        consumer_leader_var: str
        scheduler_leader_predicate: str
        tile_sched_params_var: str
        tile_sched_var: str
        work_tile_var: str
        work_tile_smem_ptr: str
        work_tile_smem: str
        work_tile_smem_tensor: str
        work_tile_coord_vars: list[str]
        work_tile_valid_var: str
        linear_pid_expr: str
        sched_pipeline_mbars: str
        sched_pipeline: str
        sched_pipeline_producer_group: str
        sched_pipeline_consumer_group: str
        sched_producer_state: str
        sched_consumer_state: str
        sched_barrier_ptr: str
        sched_peer_rank: str
        sched_peer_m: str
        refresh_work_tile_stmts: list[ast.stmt]
        work_tile_publish_stmts: list[ast.stmt]
        work_tile_consume_stmts: list[ast.stmt]
        work_tile_release_stmts: list[ast.stmt]

    def _build_tcgen05_persistent_layout(
        self, device_function: DeviceFunction
    ) -> _Tcgen05PersistentLayout:
        """Allocate persistent-kernel variables and build the work-tile
        publish/consume/release/refresh statement helpers shared between
        the prelude and the per-tile body.
        """
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
        consumer_leader_var = device_function.new_var("tcgen05_sched_consumer_leader")
        scheduler_leader_predicate = (
            cluster_scheduler_leader if cluster_m > 1 else scheduler_owner_warp
        )
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

        refresh_work_tile: list[ast.stmt] = [
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
            work_tile_publish: list[ast.stmt] = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                # The scheduler pipeline is intentionally one-stage. Match
                # Quack by arming the consumer-state full mbarrier before the
                # async remote stores; if this pipeline gains more stages,
                # producer/consumer state pairing must be revisited together.
                statement_from_string(
                    f"{sched_barrier_ptr} = {sched_pipeline}.producer_get_barrier({sched_consumer_state})"
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
                        # _cute_store_shared_remote_x4 writes four Int32
                        # values, so each remote async transaction expects
                        # 16 bytes.
                        statement_from_string(
                            "cute.arch.mbarrier_arrive_and_expect_tx("
                            f"{sched_barrier_ptr}, 16, {sched_peer_rank})"
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
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
            ]
            work_tile_consume: list[ast.stmt] = [
                statement_from_string(
                    f"{sched_pipeline}.consumer_wait({sched_consumer_state})"
                ),
                statement_from_string("cute.arch.fence_view_async_shared()"),
                statement_from_string("cute.arch.sync_warp()"),
            ]
            work_tile_release: list[ast.stmt] = [
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

        return self._Tcgen05PersistentLayout(
            cluster_m=cluster_m,
            scheduler_owner_warp=scheduler_owner_warp,
            cluster_scheduler_leader=cluster_scheduler_leader,
            consumer_leader_var=consumer_leader_var,
            scheduler_leader_predicate=scheduler_leader_predicate,
            tile_sched_params_var=tile_sched_params_var,
            tile_sched_var=tile_sched_var,
            work_tile_var=work_tile_var,
            work_tile_smem_ptr=work_tile_smem_ptr,
            work_tile_smem=work_tile_smem,
            work_tile_smem_tensor=work_tile_smem_tensor,
            work_tile_coord_vars=work_tile_coord_vars,
            work_tile_valid_var=work_tile_valid_var,
            linear_pid_expr=linear_pid_expr,
            sched_pipeline_mbars=sched_pipeline_mbars,
            sched_pipeline=sched_pipeline,
            sched_pipeline_producer_group=sched_pipeline_producer_group,
            sched_pipeline_consumer_group=sched_pipeline_consumer_group,
            sched_producer_state=sched_producer_state,
            sched_consumer_state=sched_consumer_state,
            sched_barrier_ptr=sched_barrier_ptr,
            sched_peer_rank=sched_peer_rank,
            sched_peer_m=sched_peer_m,
            refresh_work_tile_stmts=refresh_work_tile,
            work_tile_publish_stmts=work_tile_publish,
            work_tile_consume_stmts=work_tile_consume,
            work_tile_release_stmts=work_tile_release,
        )

    def _build_tcgen05_persistent_prelude(
        self, layout: _Tcgen05PersistentLayout
    ) -> list[ast.stmt]:
        """Pre-loop init: allocate SMEM, set up the tile scheduler, fetch
        the initial work tile, and publish/consume it so every warp sees
        a coherent first tile.
        """
        prelude: list[ast.stmt] = [
            statement_from_string(
                f"{layout.tile_sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                f"{self._tcgen05_num_tiles_expr(is_device=True)}, "
                f"({layout.cluster_m}, 1, 1))"
            ),
            statement_from_string(
                f"{layout.tile_sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                f"{layout.tile_sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
            ),
            statement_from_string(
                f"{layout.work_tile_smem_ptr} = cute.arch.alloc_smem(cutlass.Int32, 4, alignment=16)"
            ),
            statement_from_string(
                f"{layout.work_tile_smem_tensor} = cute.make_tensor("
                f"{layout.work_tile_smem_ptr}, cute.make_layout((4,), stride=(1,)))"
            ),
            statement_from_string(
                f"{layout.work_tile_smem} = {layout.work_tile_smem_tensor}"
            ),
        ]
        if layout.cluster_m > 1:
            prelude.extend(
                [
                    statement_from_string(
                        f"{layout.sched_pipeline_mbars} = cute.arch.alloc_smem(cutlass.Int64, cutlass.Int32(2))"
                    ),
                    # Only the scheduler leader CTA publishes each remote
                    # work tile, so every peer full barrier receives one
                    # arrive-and-expect-tx, not one arrival per cluster CTA.
                    statement_from_string(
                        f"{layout.sched_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
                        "cutlass.pipeline.Agent.Thread, 1)"
                    ),
                    statement_from_string(
                        f"{layout.sched_pipeline_consumer_group} = cutlass.pipeline.CooperativeGroup("
                        f"cutlass.pipeline.Agent.Thread, {layout.cluster_m})"
                    ),
                    statement_from_string(
                        f"{layout.sched_pipeline} = cutlass.pipeline.PipelineAsync.create("
                        "num_stages=1, "
                        f"producer_group={layout.sched_pipeline_producer_group}, "
                        f"consumer_group={layout.sched_pipeline_consumer_group}, "
                        f"barrier_storage={layout.sched_pipeline_mbars}, "
                        "consumer_mask=cutlass.Int32(0), "
                        "defer_sync=True)"
                    ),
                    statement_from_string(
                        f"{layout.sched_producer_state} = cutlass.pipeline.make_pipeline_state("
                        "cutlass.pipeline.PipelineUserType.Producer, 1)"
                    ),
                    statement_from_string(
                        f"{layout.sched_consumer_state} = cutlass.pipeline.make_pipeline_state("
                        "cutlass.pipeline.PipelineUserType.Consumer, 1)"
                    ),
                    statement_from_string(
                        f"{layout.consumer_leader_var} = "
                        "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(0) "
                        "and cute.arch.lane_idx() == cutlass.Int32(0)"
                    ),
                ]
            )
        else:
            prelude.append(
                statement_from_string(f"{layout.consumer_leader_var} = False")
            )
        prelude.append(
            self._tcgen05_scheduler_if(
                layout.scheduler_leader_predicate,
                [
                    statement_from_string(
                        f"{layout.work_tile_var} = {layout.tile_sched_var}.initial_work_tile_info()"
                    ),
                    *layout.work_tile_publish_stmts,
                ],
            )
        )
        if layout.cluster_m > 1:
            prelude.append(
                self._tcgen05_scheduler_if(
                    layout.consumer_leader_var,
                    list(layout.work_tile_consume_stmts),
                )
            )
        prelude.append(statement_from_string("cute.arch.sync_threads()"))
        prelude.extend(layout.refresh_work_tile_stmts)
        if layout.cluster_m > 1:
            prelude.append(
                self._tcgen05_scheduler_if(
                    layout.consumer_leader_var,
                    list(layout.work_tile_release_stmts),
                )
            )
        return prelude

    def _emit_role_block_stmts(
        self, role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock
    ) -> list[ast.stmt]:
        """Emit a role block's statements, gated on its role predicate.

        Shared blocks (``role_predicate is None``) emit naked
        statements -- there is no per-warp gating, every warp runs them.
        Role-gated blocks wrap their statements in ``if {predicate}:``
        so only the matching warps execute the body. An empty
        non-shared block emits nothing (no degenerate ``if {}:``).
        """
        if not role_block.stmts:
            return []
        if role_block.role_predicate is None:
            return list(role_block.stmts)
        return [
            create(
                ast.If,
                test=expr_from_string(role_block.role_predicate),
                body=list(role_block.stmts),
                orelse=[],
            )
        ]

    def _build_tcgen05_persistent_tile_body(
        self,
        layout: _Tcgen05PersistentLayout,
        role_blocks: list[Tcgen05PersistentProgramIDs._PersistentRoleBlock],
        *,
        emit_block_wide_sync: bool = True,
    ) -> list[ast.stmt]:
        """Per-tile body inside the single shared ``while``: run the
        user's kernel body (split into warp-role blocks), then advance
        the scheduler and refresh the published work tile so the next
        iteration sees the updated state.

        Role blocks are emitted in the order returned by
        ``_collect_tcgen05_role_blocks``, which preserves the original
        emit order of the per-tile body. TMA-load role blocks become
        ``if {tma_warp_predicate}: ...`` wrappers in place of the
        original tagged statements; shared blocks emit naked
        statements. The defines-before-uses invariant from the
        pre-split body carries through, so single-tile correctness is
        unchanged. Multi-tile remains gated by the host-side guard when
        this shared-only shape is used for the legacy non-role-local path.
        Static full-tile role-local kernels use sibling role-local loops
        and lift that guard only for validated ``cluster_m == 1`` configs.

        ``emit_block_wide_sync`` controls the per-tile
        ``cute.arch.sync_threads()`` (a CTA-wide barrier). The default
        ``True`` is correct for the current mainloop-role-local
        intermediate because every warp still enters this shared
        ``while`` after any role-local mainloop work. Passing ``False``
        is reserved for the later fully role-local shape where no
        role-local warp reaches the shared loop and the remaining work
        has a replacement non-CTA synchronization scheme.

        See :meth:`_build_tcgen05_persistent_tile_body_role_local` for
        the role-local-while consumer that lifts non-shared role blocks
        into sibling ``while`` loops.
        """
        body: list[ast.stmt] = [
            statement_from_string(f"{self.virtual_pid_var} = {layout.linear_pid_expr}"),
        ]
        for role_block in role_blocks:
            body.extend(self._emit_role_block_stmts(role_block))
        body.append(
            self._tcgen05_scheduler_if(
                layout.scheduler_leader_predicate,
                [
                    statement_from_string(
                        f"{layout.tile_sched_var}.advance_to_next_work()"
                    ),
                    statement_from_string(
                        f"{layout.work_tile_var} = {layout.tile_sched_var}.get_current_work()"
                    ),
                    *layout.work_tile_publish_stmts,
                ],
            )
        )
        if layout.cluster_m > 1:
            body.append(
                self._tcgen05_scheduler_if(
                    layout.consumer_leader_var,
                    list(layout.work_tile_consume_stmts),
                )
            )
        if emit_block_wide_sync:
            body.append(statement_from_string("cute.arch.sync_threads()"))
        body.extend(layout.refresh_work_tile_stmts)
        if layout.cluster_m > 1:
            body.append(
                self._tcgen05_scheduler_if(
                    layout.consumer_leader_var,
                    list(layout.work_tile_release_stmts),
                )
            )
        return body

    def _build_role_local_while(
        self,
        device_function: DeviceFunction,
        layout: _Tcgen05PersistentLayout,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None = None,
    ) -> ast.stmt:
        """Build a role-local ``while`` for one extracted role block.

        Each role-local ``while`` carries its own ``StaticPersistentTileScheduler``
        instance constructed with the same cluster shape as the shared
        scheduler (``(layout.cluster_m, 1, 1)``) so the role-local
        scheduler iterates exactly the same tile sequence in the same
        order. The role-local loop body runs the role's statements once
        per tile, advances its own scheduler, and refreshes its own
        work-tile state.

        Cross-role producer-consumer synchronization is via the AB /
        acc pipelines (the existing pipeline barriers carry the data
        dependency); no ``cute.arch.sync_threads()`` is emitted inside
        the role-local loop. The role-local warp still enters the shared
        loop afterwards so the shared loop's existing CTA-wide barriers
        remain valid.

        The returned statement is the role-local ``while`` itself,
        wrapped in ``if {role_predicate}:`` so only the matching warps
        enter the loop. The caller appends this statement as a sibling
        of the shared ``while`` inside the persistent kernel's setup
        list.

        ``scheduler_var_prefix`` selects the prefix for every variable
        name allocated in the role-local while (e.g.
        ``f"{prefix}_tile_sched"``). The caller threads a unique prefix
        per role so two role-local whiles do not collide on the same
        ``DeviceFunction.new_var`` namespace.
        """
        assert role_block.role_predicate is not None, (
            "_build_role_local_while requires a non-shared role block; "
            "shared blocks live in the shared while"
        )
        # Match the shared scheduler's cluster shape so the role-local
        # scheduler visits the same tile sequence in the same order.
        # The shared scheduler uses (layout.cluster_m, 1, 1); diverging
        # here would re-order tiles and break AB-pipeline ordering
        # between the TMA-load warp and the consumer warps.
        sched_params_var = device_function.new_var(
            f"{scheduler_var_prefix}_tile_sched_params"
        )
        sched_var = device_function.new_var(f"{scheduler_var_prefix}_tile_sched")
        work_tile_var = device_function.new_var(f"{scheduler_var_prefix}_work_tile")

        prelude: list[ast.stmt] = [
            statement_from_string(
                f"{sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                f"{self._tcgen05_num_tiles_expr(is_device=True)}, "
                f"({layout.cluster_m}, 1, 1))"
            ),
            statement_from_string(
                f"{sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                f"{sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
            ),
            statement_from_string(
                f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
            ),
        ]
        tile_counter_var = None
        if (
            role_block.role_predicate == self._tcgen05_epi_role_predicate()
            and device_function.cute_tcgen05_epi_role_tile_counter_var is not None
        ):
            tile_counter_var = device_function.cute_tcgen05_epi_role_tile_counter_var
            prelude.append(
                statement_from_string(f"{tile_counter_var} = cutlass.Int32(0)")
            )

        # Per-iteration refresh of role-local work-tile coordinates.
        # The role block's statements reference ``self.virtual_pid_var``
        # transitively (through PID decomposition), so before running
        # the role block we bind virtual_pid_var to the linearized
        # coordinate of THIS role-local work tile. The role-local
        # scheduler shares its cluster shape with the shared scheduler,
        # so the two iterate the same tiles in the same order and the
        # role-local virtual_pid_var matches the shared one tile-by-tile.
        coord_terms: list[str] = []
        for i in range(len(self.pid_info)):
            coord_terms.append(f"{work_tile_var}.tile_idx[{i}]")
        linear_pid_expr = self._tcgen05_linear_virtual_pid_from_coords_expr(coord_terms)

        per_tile_body: list[ast.stmt] = [
            statement_from_string(f"{self.virtual_pid_var} = {linear_pid_expr}"),
        ]
        if dependency_stmts is not None:
            per_tile_body.extend(dependency_stmts)
        per_tile_body.extend(role_block.stmts)
        if tile_counter_var is not None:
            per_tile_body.append(
                statement_from_string(
                    f"{tile_counter_var} = {tile_counter_var} + cutlass.Int32(1)"
                )
            )
        per_tile_body.extend(
            [
                statement_from_string(f"{sched_var}.advance_to_next_work()"),
                statement_from_string(
                    f"{work_tile_var} = {sched_var}.get_current_work()"
                ),
            ]
        )

        prelude.append(
            create(
                ast.While,
                test=expr_from_string(f"{work_tile_var}.is_valid_tile"),
                body=per_tile_body,
                orelse=[],
            )
        )

        return create(
            ast.If,
            test=expr_from_string(role_block.role_predicate),
            body=prelude,
            orelse=[],
        )

    def _role_local_dependency_stmts(
        self, shared_body: list[ast.stmt], role_stmts: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Return shared per-tile statements needed by an extracted role.

        Extracted TMA-load statements still read tile-local names such as
        ``offset_0`` / ``offset_1`` that are normally produced by the shared
        PID-decomposition prefix. Walk the shared body backwards from the
        role's reads and pull in the nearest definitions, adding their reads
        transitively. The returned statements preserve source order and run
        immediately after the role-local ``virtual_pid`` binding.

        This intentionally simple pass assumes the dependency prefix is made
        of flat, unconditional per-tile assignments (PID decomposition,
        offsets, TMA tensor partitions). ``ast.walk`` treats writes inside
        compound statements as unconditional; if conditional prefix defines
        become necessary, this helper needs control-flow-aware dominance.
        """
        needed: set[str] = set()
        internal_writes: set[str] = set()
        for stmt in role_stmts:
            reads, writes = _stmt_name_uses(stmt)
            needed.update(reads)
            internal_writes.update(writes)
        needed.difference_update(internal_writes)

        selected_reversed: list[ast.stmt] = []
        for stmt in reversed(shared_body):
            reads, writes = _stmt_name_uses(stmt)
            if not writes or not (writes & needed):
                continue
            selected_reversed.append(stmt)
            needed.difference_update(writes)
            needed.update(reads)
        selected_reversed.reverse()
        return selected_reversed

    def _build_tcgen05_persistent_tile_body_role_local(
        self,
        device_function: DeviceFunction,
        layout: _Tcgen05PersistentLayout,
        partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody,
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """Build the per-tile body in role-local-while form.

        Returns ``(role_local_whiles, shared_tile_body)`` where:

        - ``role_local_whiles`` is a list of role-local ``while`` siblings
          -- one per unique ``role_predicate`` in
          ``partition.role_blocks_extracted``. Multiple extracted role
          blocks sharing the same predicate are merged into a single
          role-local loop with their statements concatenated in the
          order they appear in the source body, so per-tile ordering
          across the role's statements is preserved (otherwise tile 0's
          first chunk would run for every tile before tile 0's second
          chunk ran, breaking the AB-pipeline ordering). Each loop is
          wrapped in ``if {role_predicate}:`` so only the matching
          warps enter.
        - ``shared_tile_body`` is the per-tile body for the shared
          ``while`` (the work-tile body without the extracted role
          blocks). Built via :meth:`_build_tcgen05_persistent_tile_body`
          with existing ``cute.arch.sync_threads()`` calls preserved. In
          this intermediate all warps still enter the shared ``while`` after
          any role-local mainloop work, so those CTA-wide barriers remain
          valid for epilogue synchronization and work-tile metadata
          publication.

        Caller wires both into the persistent kernel as siblings of
        each other inside the same setup list. Each role-local ``while``
        runs only on its predicated warps, then every warp enters the shared
        ``while`` so existing CTA-wide barriers remain valid.

        **Current limitation.** The TMA-load, MMA-exec, and TMA-store
        epilogue roles are extracted today. Single-root static full-tile
        multi-tile correctness is validated for ``cluster_m == 1``. Partial
        fallback shapes, multi-root ForEach kernels, and cluster_m > 1
        configs remain guarded for runtime execution, and autotune keeps
        cluster_m=2 out of the search until the G3 ownership path is
        runtime-correct and benchmarked.
        """
        # Wrap the shared body's tagged-removed view in the standard
        # per-tile shape. ``shared_role_blocks`` reuses the
        # inline-weave block structure but only over the
        # extracted-shared statements; tagged stmts have been pulled
        # out into ``role_blocks_extracted``.
        shared_role_blocks = [
            self._PersistentRoleBlock(
                role_predicate=None, stmts=list(partition.shared_body_extracted)
            )
        ]
        shared_tile_body = self._build_tcgen05_persistent_tile_body(
            layout, shared_role_blocks
        )
        # Merge extracted blocks by ``role_predicate`` so each predicate
        # gets one role-local loop carrying all of its per-tile
        # statements in source order. Emit the loops in explicit role
        # order instead of first-seen source order: TMA-load publishes
        # operands, MMA-exec consumes them and publishes accumulator stages,
        # then epi consumes those stages. Adding another role must update
        # ``role_order`` so omitted predicates fail loudly.
        merged: dict[str, list[ast.stmt]] = {}
        for role_block in partition.role_blocks_extracted:
            assert role_block.role_predicate is not None
            merged.setdefault(role_block.role_predicate, []).extend(role_block.stmts)
        role_local_whiles: list[ast.stmt] = []
        role_order = {
            self._tcgen05_tma_load_role_predicate(): 0,
            self._tcgen05_mma_exec_role_predicate(): 1,
            self._tcgen05_epi_role_predicate(): 2,
        }
        unknown_predicates = set(merged) - set(role_order)
        assert not unknown_predicates, (
            "tcgen05 role-local order missing predicate(s): "
            + ", ".join(sorted(unknown_predicates))
        )
        ordered_predicates = sorted(merged, key=lambda predicate: role_order[predicate])
        for i, predicate in enumerate(ordered_predicates):
            stmts = merged[predicate]
            merged_block = self._PersistentRoleBlock(
                role_predicate=predicate, stmts=stmts
            )
            role_local_whiles.append(
                self._build_role_local_while(
                    device_function,
                    layout,
                    merged_block,
                    scheduler_var_prefix=f"tcgen05_role_local_{i}",
                    dependency_stmts=self._role_local_dependency_stmts(
                        partition.shared_body_extracted, stmts
                    ),
                )
            )
        return role_local_whiles, shared_tile_body

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        return self._setup_tcgen05_persistent_kernel(device_function)
