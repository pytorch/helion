"""One physical grid/phase stride, checked against the executing CUDA function."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...exc import InvalidConfig
from ..ast_extension import create
from ..ast_extension import iter_statements
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..program_id import NUM_SM_VAR
from ..program_id import ForEachProgramID
from ..program_id import PersistentInterleavedProgramIDs
from ..program_id import ProgramIDs
from ._ast_pass_utils import phase_barrier_call
from .autotune.config_keys import STAGED_GRID_KEY
from .autotune.staged_projection_grid import staged_task_bound
from .state import cute_codegen_state

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


class StagedProjectionProgramIDs(PersistentInterleavedProgramIDs):
    """Tile-count-capped interleaving; all phases consume the same constexpr."""

    def __init__(self, fn: DeviceFunction) -> None:
        env = CompileEnvironment.current()
        estimated = staged_task_bound(env.config_spec, fn.config)
        waves = fn.config.get(STAGED_GRID_KEY)
        if (
            not estimated
            or type(waves) is not int
            or waves not in (2, 3)
            or env.settings.persistent_reserved_sms
        ):
            raise InvalidConfig("unsupported staged projection grid geometry")
        self.waves = waves
        self.task_limit: ast.Constant | None = None
        self.installed_tasks: int | None = None
        super().__init__()

    def physical_grid_statement(self, reserved_arg: str) -> ast.stmt:
        # The first PID case owns this shared constexpr. Bind its cap only
        # after the actual resident stages, including fallbacks, are known.
        self.task_limit = create(ast.Constant, value=0)
        return statement_from_string(
            f"{NUM_SM_VAR} = min({{limit}}, helion.runtime.get_num_sm({self.get_device_str()}{reserved_arg}) * {self.waves})",
            limit=self.task_limit,
        )

    def static_grid_size(self) -> int | None:
        sms = super().static_grid_size()
        if self.installed_tasks is None or sms is None:
            # Do not specialize another transform from a merely potential root.
            return None
        return min(self.installed_tasks, sms * self.waves)


def validate_staged_grid(
    fn: DeviceFunction, definition: ast.FunctionDef
) -> tuple[int, dict[ast.stmt, ast.Call]]:
    """Require one real staged domain and unconditional local joins before export."""
    ir = HostFunction.current().device_ir
    schedule = fn.pid
    if (
        not isinstance(schedule, ForEachProgramID)
        or len(schedule.cases) != len(ir.root_ids)
        or (len(set(schedule.case_phases)) < 2 and not ir.implicit_dependency_starts)
        or cute_codegen_state(fn.codegen).wrapper_plans
        or fn.cute_state.lanes.simt_cluster_n != 1
        or fn.cute_state.lanes.grid_cluster_n != 1
    ):
        raise InvalidConfig("checked staged grid requires ordinary multi-root phases")
    joins = [
        (s, call)
        for s in iter_statements(definition.body)
        if (call := phase_barrier_call(s)) is not None
    ]
    if (
        any(isinstance(n, ast.Return) for n in ast.walk(definition))
        or not joins
        or any(
            stmt not in definition.body
            or not isinstance(call.func, ast.Name)
            or call.func.id != "_cute_grid_barrier"
            for stmt, call in joins
        )
    ):
        raise InvalidConfig(
            "checked staged grid requires unconditional local grid joins"
        )
    installed = fn.cute_state.installed_staged_roots
    counts: list[int] = []
    cases: list[StagedProjectionProgramIDs] = []
    for root, case in zip(ir.root_ids, schedule.cases, strict=True):
        if not isinstance(case, StagedProjectionProgramIDs):
            raise InvalidConfig("all phase strides must use the checked staged grid")
        cases.append(case)
        if root in installed:
            # The base implementation counts the actual logical PID axes.
            # The overridden static_grid_size instead reports physical CTAs.
            count = ProgramIDs.static_grid_size(case)
            if count is None:
                raise InvalidConfig("dynamic installed projection task domain")
            counts.append(count)
    tasks = max(counts, default=0)
    if not CompileEnvironment.current().config_spec.num_sm < tasks < 2**31:
        raise InvalidConfig("no installed staged projection requires a wider grid")
    limits = [case.task_limit for case in cases if case.task_limit is not None]
    if len(limits) != 1:
        raise InvalidConfig("checked phase grid needs exactly one host stride")
    limits[0].value = tasks
    for case in cases:
        case.installed_tasks = tasks
    args = [a.arg for a in (*definition.args.posonlyargs, *definition.args.args)]
    if args.count(NUM_SM_VAR) != 1:
        raise InvalidConfig(
            "checked staged grid requires the shared phase-stride constexpr"
        )
    return args.index(NUM_SM_VAR), dict(joins)
