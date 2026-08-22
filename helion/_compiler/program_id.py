from __future__ import annotations

import abc
import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NamedTuple
from typing import cast

import torch

from .. import exc
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.cutedsl_compat import emit_pipeline_advance
from .cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_DEFAULT
from .cute.strategies import l2_swizzle_size_from_config
from .cute.tcgen05_constants import TCGEN05_GROUPED_STATIC_SPECIALIZATION_MAX_GROUPS
from .cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT
from .cute.tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .cute.tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .cute.tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER
from .cute.tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .device_function import DeviceFunction
from .device_function import TensorArg
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .tile_dependency_planner import AccessCohortPlan
from .tile_dependency_planner import AccessProgramPoint
from .tile_dependency_planner import InstantiatedTaskFamily
from .tile_dependency_planner import TaskContinuationPipelinePlan
from .tile_dependency_planner import TaskContinuationPlan
from .tile_dependency_planner import build_generic_schedule_plan


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
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            reads.add(node.target.id)
    return reads, writes


def _clone_ast_value(value: object) -> object:
    if isinstance(value, list):
        from .cross_loop_dependencies import cross_loop_access_marker_id

        return [
            _clone_ast_value(item)
            for item in value
            if not (
                isinstance(item, ast.AST)
                and cross_loop_access_marker_id(item) is not None
            )
        ]
    if isinstance(value, tuple):
        return tuple(_clone_ast_value(item) for item in value)
    if isinstance(value, ast.AST):
        from .cross_loop_dependencies import CROSS_LOOP_ACCESS_ID_META
        from .cross_loop_dependencies import cross_loop_access_marker_id

        access_id = cross_loop_access_marker_id(value)
        fields = {
            field: _clone_ast_value(getattr(value, field)) for field in value._fields
        }
        if isinstance(value, ExtendedAST):
            cloned = value.copy(**fields)
        else:
            cloned = ast.copy_location(type(value)(**fields), value)
        if access_id is not None:
            setattr(cloned, CROSS_LOOP_ACCESS_ID_META, access_id)
        return cloned
    return value


def _clone_stmt(stmt: ast.stmt) -> ast.stmt:
    return cast("ast.stmt", _clone_ast_value(stmt))


def _ast_fingerprint(nodes: list[ast.stmt]) -> tuple[str, ...]:
    """Return a location-independent fingerprint for an opaque computation body."""
    from .cross_loop_dependencies import cross_loop_access_marker_id

    return tuple(
        ast.dump(cast("ast.AST", _clone_ast_value(node)), include_attributes=False)
        for node in nodes
        if cross_loop_access_marker_id(node) is None
    )


def _clone_opaque_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Clone a tile body while proving that no computation was rewritten."""
    cloned = [_clone_stmt(statement) for statement in body]
    if _ast_fingerprint(cloned) != _ast_fingerprint(body):
        raise AssertionError("opaque tile-body cloning changed its computation")
    return cloned


def _clone_opaque_statements_with_split_access_loops(
    body: list[ast.stmt],
    split_offsets_by_access: dict[int, str],
) -> list[ast.stmt]:
    """Split loops containing selected accesses without changing their bodies."""
    from .cross_loop_dependencies import cross_loop_access_marker_id

    split_accesses: set[int] = set()

    def direct_marker_ids(value: object) -> list[int]:
        result: list[int] = []

        def visit(item: object) -> None:
            if isinstance(item, list | tuple):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, ast.AST):
                return
            if (access_id := cross_loop_access_marker_id(item)) is not None:
                if access_id in split_offsets_by_access and access_id not in result:
                    result.append(access_id)
                return
            if isinstance(item, ast.For):
                return
            for field in item._fields:
                visit(getattr(item, field))

        visit(value)
        return result

    def clone(value: object) -> object:
        if isinstance(value, list):
            result: list[object] = []
            for item in value:
                if (
                    isinstance(item, ast.AST)
                    and cross_loop_access_marker_id(item) is not None
                ):
                    continue
                if isinstance(item, ast.For) and (
                    access_ids := direct_marker_ids(item.body)
                ):
                    split_offsets = {
                        split_offsets_by_access[access_id] for access_id in access_ids
                    }
                    if len(split_offsets) != 1:
                        raise AssertionError(
                            "one consumer loop cannot use incompatible "
                            "cross-loop split offsets"
                        )
                    split = cast("ast.expr", expr_from_string(split_offsets.pop()))
                    result.extend(
                        (
                            _clone_opaque_loop_segment(item, end=split),
                            _clone_opaque_loop_segment(item, begin=split),
                        )
                    )
                    split_accesses.update(access_ids)
                    continue
                result.append(clone(item))
            return result
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, ast.AST):
            fields = {field: clone(getattr(value, field)) for field in value._fields}
            if isinstance(value, ExtendedAST):
                return value.copy(**fields)
            return ast.copy_location(type(value)(**fields), value)
        return value

    cloned = cast("list[ast.stmt]", clone(body))
    if split_accesses != split_offsets_by_access.keys():
        missing = sorted(split_offsets_by_access.keys() - split_accesses)
        raise AssertionError(f"missing cross-loop access loops: {missing}")
    return cloned


def _clone_opaque_statements_with_access_stages(
    body: list[ast.stmt],
    *,
    access_ids: frozenset[int],
    split_offsets: tuple[str, ...],
    stage_waits: tuple[tuple[ast.stmt, ...], ...],
) -> list[ast.stmt]:
    """Split the loop owning ``access_ids`` and wait before each segment."""
    from .cross_loop_dependencies import cross_loop_access_marker_id

    if len(stage_waits) != len(split_offsets) + 1:
        raise AssertionError("each access-loop segment requires one wait")
    scheduled = False

    def contains_access(value: object) -> bool:
        if isinstance(value, list | tuple):
            return any(contains_access(item) for item in value)
        if not isinstance(value, ast.AST):
            return False
        access_id = cross_loop_access_marker_id(value)
        if access_id is not None:
            return access_id in access_ids
        if isinstance(value, ast.For):
            return False
        return any(contains_access(getattr(value, field)) for field in value._fields)

    def clone(value: object) -> object:
        nonlocal scheduled
        if isinstance(value, list):
            result: list[object] = []
            for item in value:
                if (
                    isinstance(item, ast.AST)
                    and cross_loop_access_marker_id(item) is not None
                ):
                    continue
                if isinstance(item, ast.For) and contains_access(item.body):
                    if scheduled:
                        raise AssertionError(
                            "one coarsened access event must belong to one loop"
                        )
                    boundaries = (None, *split_offsets, None)
                    for index, waits in enumerate(stage_waits):
                        result.extend(_clone_opaque_statements(list(waits)))
                        begin_text = boundaries[index]
                        end_text = boundaries[index + 1]
                        begin = (
                            cast("ast.expr", expr_from_string(begin_text))
                            if begin_text is not None
                            else None
                        )
                        end = (
                            cast("ast.expr", expr_from_string(end_text))
                            if end_text is not None
                            else None
                        )
                        result.append(
                            _clone_opaque_loop_segment(item, begin=begin, end=end)
                        )
                    scheduled = True
                    continue
                result.append(clone(item))
            return result
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, ast.AST):
            fields = {field: clone(getattr(value, field)) for field in value._fields}
            if isinstance(value, ExtendedAST):
                return value.copy(**fields)
            return ast.copy_location(type(value)(**fields), value)
        return value

    cloned = cast("list[ast.stmt]", clone(body))
    if not scheduled:
        raise AssertionError("missing coarsened cross-loop consumer loop")
    return cloned


_CROSS_LOOP_PROGRAM_POINT_LOOP_ID = "_cross_loop_program_point_loop_id"


def _clone_opaque_statements_with_access_iteration_wait(
    body: list[ast.stmt],
    *,
    access_ids: frozenset[int],
    loop_id: int,
    wait: list[ast.stmt],
) -> list[ast.stmt]:
    """Wait once per iteration of the loop containing related accesses."""
    from .cross_loop_dependencies import cross_loop_access_marker_id

    scheduled = False

    def clone(value: object) -> object:
        nonlocal scheduled
        if isinstance(value, list):
            result: list[object] = []
            for item in value:
                if (
                    isinstance(item, ast.AST)
                    and cross_loop_access_marker_id(item) is not None
                ):
                    continue
                result.append(clone(item))
            return result
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if not isinstance(value, ast.AST):
            return value
        if (
            isinstance(value, ast.For)
            and getattr(value, _CROSS_LOOP_PROGRAM_POINT_LOOP_ID, None) == loop_id
        ):
            if scheduled:
                raise AssertionError("cross-loop program point loop is not unique")
            present_access_ids = {
                access_id
                for item in ast.walk(value)
                if (access_id := cross_loop_access_marker_id(item)) is not None
            }
            if not access_ids <= present_access_ids:
                raise AssertionError(
                    "cross-loop access markers left their program point"
                )
            scheduled = True
            return _prepend_schedule_to_opaque_loop(
                value,
                wait,
                force_serial_pipeline=True,
            )
        fields = {field: clone(getattr(value, field)) for field in value._fields}
        if isinstance(value, ExtendedAST):
            return value.copy(**fields)
        return ast.copy_location(type(value)(**fields), value)

    cloned = cast("list[ast.stmt]", clone(body))
    if not scheduled:
        raise AssertionError("missing cross-loop access program point loop")
    return cloned


def _collect_cross_loop_access_program_points(
    case_bodies: list[list[ast.stmt]],
    access_coordinates: dict[int, dict[int, str] | None],
) -> tuple[tuple[frozenset[int], ...], dict[int, AccessProgramPoint]]:
    """Bind explicit access markers to their lowered logical coordinates."""
    from .cross_loop_dependencies import cross_loop_access_marker_id

    available_by_root: list[set[int]] = [set() for _ in case_bodies]
    program_points: dict[int, AccessProgramPoint] = {}
    ambiguous_accesses: set[int] = set()
    next_loop_id = 0

    def visit(
        value: object,
        root: int,
        loop_stack: tuple[int, ...],
        root_statement_index: int,
    ) -> None:
        nonlocal next_loop_id
        if isinstance(value, list | tuple):
            for item in value:
                visit(item, root, loop_stack, root_statement_index)
            return
        if not isinstance(value, ast.AST):
            return
        if (access_id := cross_loop_access_marker_id(value)) is not None:
            available_by_root[root].add(access_id)
            coordinates = access_coordinates.get(access_id)
            point = AccessProgramPoint(
                access_id=access_id,
                coordinate_items=(
                    tuple(sorted(coordinates.items()))
                    if coordinates is not None
                    else None
                ),
                loop_id=loop_stack[-1] if loop_stack else None,
                loop_depth=len(loop_stack),
                root_statement_index=root_statement_index,
            )
            if (previous := program_points.get(access_id)) is not None:
                if previous != point:
                    ambiguous_accesses.add(access_id)
            else:
                program_points[access_id] = point
            return
        if isinstance(value, ast.For):
            loop_id = next_loop_id
            next_loop_id += 1
            setattr(value, _CROSS_LOOP_PROGRAM_POINT_LOOP_ID, loop_id)
            loop_stack = (*loop_stack, loop_id)
        for field in value._fields:
            visit(getattr(value, field), root, loop_stack, root_statement_index)

    for root, body in enumerate(case_bodies):
        for root_statement_index, statement in enumerate(body):
            visit(statement, root, (), root_statement_index)
    for access_id in ambiguous_accesses:
        program_points.pop(access_id, None)
    return tuple(frozenset(ids) for ids in available_by_root), program_points


def _clone_opaque_loop_segment(
    loop: ast.For,
    *,
    begin: ast.expr | None = None,
    end: ast.expr | None = None,
) -> ast.For:
    """Clone one existing loop, changing only scheduling range boundaries.

    The loop body is deliberately treated as opaque.  Splitting an existing
    ordered loop is permitted for overlap, but its arithmetic, stores, and
    statement order must remain identical in every segment.
    """
    cloned = cast("ast.For", _clone_ast_value(loop))
    if not (
        isinstance(cloned.iter, ast.Call)
        and isinstance(loop.iter, ast.Call)
        and len(cloned.iter.args) >= 2
    ):
        raise AssertionError("tile-dependency stages require a range-like loop")
    if begin is not None:
        cloned.iter.args[0] = begin
    if end is not None:
        cloned.iter.args[1] = end
    if _ast_fingerprint(cloned.body) != _ast_fingerprint(loop.body):
        raise AssertionError("tile-dependency staging changed an opaque loop body")
    return cloned


def _prepend_schedule_to_opaque_loop(
    loop: ast.For,
    schedule: list[ast.stmt],
    *,
    force_serial_pipeline: bool,
) -> ast.For:
    """Add synchronization before iterations without changing their computation."""
    cloned = cast("ast.For", _clone_ast_value(loop))
    computation = _ast_fingerprint(cloned.body)
    cloned.body = [*_clone_opaque_statements(schedule), *cloned.body]
    if _ast_fingerprint(cloned.body[len(schedule) :]) != computation:
        raise AssertionError("tile-dependency wait changed an opaque loop body")
    if force_serial_pipeline:
        if not isinstance(cloned.iter, ast.Call):
            raise AssertionError("ordered tile dependency requires a range loop")
        cloned.iter.keywords = [
            keyword
            for keyword in cloned.iter.keywords
            if keyword.arg not in {"num_stages", "disallow_acc_multi_buffer"}
        ]
        cloned.iter.keywords.extend(
            [
                create(
                    ast.keyword,
                    arg="num_stages",
                    value=create(ast.Constant, value=1),
                ),
                create(
                    ast.keyword,
                    arg="disallow_acc_multi_buffer",
                    value=create(ast.Constant, value=True),
                ),
            ]
        )
    return cloned


_TCGEN05_WORK_TILE_MAILBOX_VALID = 3


def _build_sched_pipeline_consumer_wait_block(
    *,
    sched_pipeline: str,
    sched_consumer_state: str,
    work_tile_smem: str,
    valid_var: str,
    valid_slot_index: int = _TCGEN05_WORK_TILE_MAILBOX_VALID,
    work_tile_stage_index: str | None = None,
) -> list[ast.stmt]:
    """Emit the consumer-side wait block for the ``ROLE_LOCAL_WITH_SCHEDULER``
    sched_pipeline: ``consumer_wait`` → ``fence_view_async_shared``
    → ``sync_warp`` → read the work-tile valid flag.

    Shared between ``_build_role_local_while_with_scheduler`` (the
    TMA-load / MMA-exec / epi consumer roles) and
    ``_build_c_input_warp_role_local_while`` (the C-input warp role
    introduced in ``cute_plan.md`` §7.5.3.2's producer-body split).
    Each call site supplies a fresh ``valid_var`` so the per-role
    valid flag has its own SMEM name; the other three arguments
    are pipeline-level and identical across roles.

    ``mbarrier.wait`` PTX stalls the issuing thread until the phase
    flips, so all 32 threads in the warp can call ``consumer_wait``
    safely (no lane-0 gate needed). The async-shared fence
    serializes the scheduler-warp's SMEM writes against the
    consumer's proxy view of SMEM, and ``sync_warp`` keeps the warp
    lanes consistent before they read the valid flag from SMEM.

    Each call returns *fresh* AST nodes — caller-supplied factory
    pattern (insertion-point-specific copies of the same shape)
    so downstream AST passes are not corrupted by sharing nodes
    across multiple parents.

    Diagnostic ``tcgen05_sched_consumer_wait_mode="warp_leader"``
    instead gates ``consumer_wait`` to lane 0 and reconverges the warp
    before the async-shared fence. This is a profiling-only wait topology
    experiment; the normal whole-warp wait path remains the default because
    B200 timing showed the lane-0 variant is slower.
    """
    try:
        wait_mode = DeviceFunction.current().config.get(
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL,
        )
    except NoCurrentFunction:
        wait_mode = TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
    valid_slot = (
        f"{work_tile_smem}[cutlass.Int32({valid_slot_index})]"
        if work_tile_stage_index is None
        else (
            f"{work_tile_smem}[cutlass.Int32({valid_slot_index}), "
            f"{work_tile_stage_index}]"
        )
    )
    if wait_mode == TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER:
        return [
            create(
                ast.If,
                test=expr_from_string("cute.arch.lane_idx() == cutlass.Int32(0)"),
                body=[
                    statement_from_string(
                        f"{sched_pipeline}.consumer_wait({sched_consumer_state})"
                    ),
                ],
                orelse=[],
            ),
            statement_from_string("cute.arch.sync_warp()"),
            statement_from_string("cute.arch.fence_view_async_shared()"),
            statement_from_string("cute.arch.sync_warp()"),
            statement_from_string(f"{valid_var} = {valid_slot} != cutlass.Int32(0)"),
        ]
    return [
        statement_from_string(
            f"{sched_pipeline}.consumer_wait({sched_consumer_state})"
        ),
        statement_from_string("cute.arch.fence_view_async_shared()"),
        statement_from_string("cute.arch.sync_warp()"),
        statement_from_string(f"{valid_var} = {valid_slot} != cutlass.Int32(0)"),
    ]


def _build_sched_pipeline_consumer_release_block(
    *,
    sched_pipeline: str,
    sched_consumer_state: str,
) -> list[ast.stmt]:
    """Emit the consumer-side release block for the
    ``ROLE_LOCAL_WITH_SCHEDULER`` sched_pipeline: lane-0-gated
    ``consumer_release`` → ``advance_state`` → ``sync_warp``.

    Companion to ``_build_sched_pipeline_consumer_wait_block``.
    ``consumer_release`` is gated on ``lane_idx == 0`` because the
    per-CTA sched-pipeline empty barrier is initialized with one
    arrival per consumer *warp* (not per-thread) — see
    ``cute_mma._codegen_cute_mma``'s
    ``consumer_mask_to_leader=False`` branch. The ``sync_warp``
    after the advance keeps the warp lanes' view of the
    register-resident consumer state consistent.
    """
    return [
        create(
            ast.If,
            test=expr_from_string("cute.arch.lane_idx() == cutlass.Int32(0)"),
            body=[
                statement_from_string(
                    f"{sched_pipeline}.consumer_release({sched_consumer_state})"
                ),
            ],
            orelse=[],
        ),
        statement_from_string(emit_pipeline_advance(sched_consumer_state)),
        statement_from_string("cute.arch.sync_warp()"),
    ]


_TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_M = 0
_TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_N = 1
_TCGEN05_GROUPED_SELECTED_MAILBOX_VALID = 2
_TCGEN05_GROUPED_SELECTED_MAILBOX_METADATA_IDX = 3
_TCGEN05_GROUPED_SELECTED_MAILBOX_GROUP_IDX = 4
_TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_M = 5
_TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_N = 6
_TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_K = 7
_TCGEN05_GROUPED_SELECTED_MAILBOX_GLOBAL_M_START = 8
if TYPE_CHECKING:
    import sympy

    from .cross_loop_dependencies import WaitSpec
    from .cute.cute_mma import _Tcgen05SchedPipelinePlan
    from .cute.device_state import CuteTcgen05MatmulPlan
    from .inductor_lowering import CodegenState

NUM_SM_VAR = "_NUM_SM"
NUM_XCD_VAR = "_NUM_XCDS"

# One 128-byte cache line in uint32 elements.  Independent broadcast counters
# are padded by this amount so polling one dependency does not invalidate a
# neighboring dependency's line.
_TILE_DEPENDENCY_COUNTER_STRIDE = 32
_TILE_DEPENDENCY_DIRECT_POLL_FANOUT_LIMIT = 100


def _partitioned_materialization_geometry(
    *,
    producer_tasks: int,
    finalized_members: int,
    finalize_partition_block: int,
    materialize_tasks: int,
) -> tuple[int, int, int, int] | None:
    """Infer a unique two-side partition geometry from static task counts.

    The materializer may expose one combined task per partition or one task
    for each of the two disjoint side values.  Returning ``None`` for an
    ambiguous or incompatible domain keeps the lowering conservative.
    """
    candidates: list[tuple[int, int, int, int]] = []
    for materialize_tasks_per_partition in (1, 2):
        if materialize_tasks % materialize_tasks_per_partition:
            continue
        group_count = materialize_tasks // materialize_tasks_per_partition
        if finalized_members <= group_count:
            continue
        primary_members = finalized_members - group_count
        if primary_members % group_count:
            continue
        primary_members_per_partition = primary_members // group_count
        total_members = finalized_members + group_count
        if producer_tasks % total_members:
            continue
        tiles_per_member = producer_tasks // total_members
        if (
            primary_members_per_partition % finalize_partition_block
            or group_count % finalize_partition_block
        ):
            continue
        candidates.append(
            (
                group_count,
                primary_members_per_partition,
                tiles_per_member,
                materialize_tasks_per_partition,
            )
        )
    return candidates[0] if len(candidates) == 1 else None


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
            "multi-phase kernels currently require persistent_blocked"
        )

        schedule = HostFunction.current().device_ir.tile_dependency_schedule
        if schedule is not None and schedule.uses_tile_dependency_counters:
            return self._emit_tile_dependency_stage_loops(
                base_strategy, device_function, total_expr
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
            env = CompileEnvironment.current()
            block_expr = env.backend.cast_expr(
                f"({'+ '.join(blocks)})", env.index_type()
            )
            state.codegen.statements_stack[-1].insert(
                0,
                statement_from_string(f"{self.shared_pid_var} -= {block_expr}"),
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
        body: list[ast.stmt],
        device_function: DeviceFunction,
        virtual_pid_var: str,
    ) -> list[ast.stmt]:
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
            cast("list[ast.stmt]", device_function.body),
            device_function,
            strategy.virtual_pid_var,
        )
        # Access markers are compiler-only insertion points. This conservative
        # phase-barrier path does not consume them, so remove them before emit.
        base_body = cast("list[ast.stmt]", _clone_ast_value(base_body))

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

    def _emit_tile_dependency_stage_loops(
        self,
        strategy: PersistentProgramIDs,
        device_function: DeviceFunction,
        total_expr: str,
    ) -> list[ast.stmt]:
        """Emit monotonic arrival-counter phases without a cooperative launch.

        Every physical worker publishes one release arrival.  All workers
        acquire-poll the same monotonic counter before entering the next phase.
        The counter target is epoch-scaled, so fixed CUDA Graph arguments need
        neither a reset kernel nor a host-side epoch update.
        """
        static_task_counts = self._static_case_task_counts(device_function)
        if static_task_counts is None:
            CompileEnvironment.current().has_barrier = True
            return self._emit_phase_loops(strategy, device_function, total_expr)

        boundaries = self._phase_boundaries()
        worker = typed_program_id(0)
        epoch_var = device_function.new_var("tile_dependency_epoch", dce=False)
        base_body = self._prepare_persistent_body(
            cast("list[ast.stmt]", device_function.body),
            device_function,
            strategy.virtual_pid_var,
        )
        case_bodies = self._extract_case_bodies(base_body)
        opaque_case_fingerprints = tuple(_ast_fingerprint(body) for body in case_bodies)
        case_offsets: list[int] = []
        running_offset = 0
        for task_count in static_task_counts:
            case_offsets.append(running_offset)
            running_offset += task_count
        singleton_roots = self._match_opaque_singleton_roots(static_task_counts)
        partitioned_pipeline_plans = self._match_partitioned_dependency_pipeline(
            case_bodies, device_function
        )
        reduction_fanout_plans = self._match_one_wave_reduction_fanouts(
            singleton_roots, case_bodies, device_function
        )
        structural_edges: set[tuple[int, int]] = set()
        for plan in partitioned_pipeline_plans.values():
            structural_edges.update(plan.covered_edges)
        for plan in reduction_fanout_plans.values():
            structural_edges.update(plan.covered_edges)
        structural_roots = {root for edge in structural_edges for root in edge}
        schedule = HostFunction.current().device_ir.tile_dependency_schedule
        assert schedule is not None and schedule.policy is not None
        dependency_plan = HostFunction.current().device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        physical_worker_limit = CompileEnvironment.current().config_spec.num_sm * cast(
            "int", device_function.config.get("num_sm_multiplier", 1)
        )
        instantiated_task_families = self._instantiated_task_families(device_function)
        assert instantiated_task_families is not None
        assert [family.task_count for family in instantiated_task_families] == (
            static_task_counts
        )
        (
            available_access_ids_by_root,
            access_program_points,
        ) = _collect_cross_loop_access_program_points(
            case_bodies,
            device_function.cross_loop_access_coordinates,
        )
        axis_geometry = {
            block_id: geometry
            for block_id in range(len(CompileEnvironment.current().block_sizes))
            if (geometry := self._static_block_axis_geometry(block_id, device_function))
            is not None
        }
        generic_plan = build_generic_schedule_plan(
            dependency_plan=dependency_plan,
            task_families=instantiated_task_families,
            available_access_ids_by_root=available_access_ids_by_root,
            access_program_points=access_program_points,
            axis_geometry=axis_geometry,
            excluded_roots=frozenset(structural_roots),
            preordered_edges=frozenset(structural_edges),
            physical_worker_limit=physical_worker_limit,
        )
        generic_task_edges = generic_plan.task_ready_edges
        whole_value_edges = set(generic_plan.root_completion_edges)
        generic_task_waits = generic_plan.task_waits_by_root
        task_continuation_plans = generic_plan.continuations
        access_cohort_plans = generic_plan.access_cohorts
        task_continuation_pipeline_plans = generic_plan.continuation_pipelines

        self._validate_tile_dependency_plan_coverage(
            singleton_roots=singleton_roots,
            partitioned_pipeline_plans=partitioned_pipeline_plans,
            reduction_fanout_plans=reduction_fanout_plans,
            task_ready_edges=generic_task_edges,
            whole_value_edges=frozenset(whole_value_edges),
        )
        epoch_replicas = schedule.policy.epoch_replicas or 1
        if epoch_replicas & (epoch_replicas - 1):
            raise exc.TileDependencyScheduleError(
                f"epoch_replicas must be a power of two, got {epoch_replicas}"
            )
        launch_worker_limit = generic_plan.worker_limit
        if task_continuation_pipeline_plans:
            strategy.grid_size_expr = (
                f"min(({strategy.grid_size_expr}), {launch_worker_limit})"
            )

        def active_worker_count(root: int) -> int:
            assert static_task_counts is not None
            return min(static_task_counts[root], launch_worker_limit)

        whole_value_poller_count = {
            producer: max(
                active_worker_count(consumer)
                for source, consumer in whole_value_edges
                if source == producer
            )
            for producer in {source for source, _ in whole_value_edges}
        }
        whole_value_replicas = {
            producer: (
                epoch_replicas
                if pollers > _TILE_DEPENDENCY_DIRECT_POLL_FANOUT_LIMIT
                else 1
            )
            for producer, pollers in whole_value_poller_count.items()
        }
        # Until a smaller progress-critical cohort is proven for a schedule,
        # require every launched worker to fit concurrently.  This is checked
        # by the launcher using CUDA's exact occupancy calculation after ptxas.
        device_function.triton_minimum_resident_programs = strategy.grid_size_expr
        device_function.preamble.extend(
            strategy._persistent_setup_statements(total_expr)
        )
        epoch_arg = self._register_tile_dependency_state(
            device_function,
            name_hint="tile_dependency_worker_epoch",
            numel=strategy.grid_size_expr,
            dtype=torch.uint32,
            zero_init=True,
        )
        task_event_roots = sorted(
            {
                dependency_plan.event(wait.event_id).producer_root
                for waits in generic_task_waits.values()
                for wait in waits
            }
        )
        task_event_offsets: dict[int, int] = {}
        task_event_count = 0
        for root in task_event_roots:
            task_event_offsets[root] = task_event_count
            task_event_count += static_task_counts[root]
        task_event_arg: str | None = None
        if task_event_count:
            task_event_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_task_epochs",
                numel=str(task_event_count),
                dtype=torch.uint32,
                zero_init=True,
            )
        access_cohort_offsets: dict[ForEachProgramID._AccessCohort, int] = {}
        access_cohort_counter_count = 0
        for plan in access_cohort_plans:
            producer_geometry = self._static_case_geometry(
                plan.producer_root, device_function
            )
            assert producer_geometry is not None
            _, producer_counts, _ = producer_geometry
            outer_count = math.prod(
                producer_counts[block_id] for block_id in plan.outer_producer_axes
            )
            access_cohort_offsets[plan] = access_cohort_counter_count
            access_cohort_counter_count += (
                outer_count
                * plan.milestone_count
                * (1 if plan.is_per_coordinate else _TILE_DEPENDENCY_COUNTER_STRIDE)
            )
        access_cohort_arg: str | None = None
        if access_cohort_counter_count:
            access_cohort_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_cohort_arrivals",
                numel=str(access_cohort_counter_count),
                dtype=torch.uint32,
                zero_init=True,
            )
        task_continuation_offsets: dict[ForEachProgramID._TaskContinuation, int] = {}
        task_continuation_counter_count = 0
        for plan in task_continuation_plans:
            task_continuation_offsets[plan] = task_continuation_counter_count
            task_continuation_counter_count += plan.consumer_tasks
        task_continuation_arg: str | None = None
        if task_continuation_counter_count:
            task_continuation_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_continuation_arrivals",
                numel=str(task_continuation_counter_count),
                dtype=torch.uint32,
                zero_init=True,
            )
        singleton_epoch_arg: str | None = None
        singleton_indices = {
            root: index for index, root in enumerate(sorted(singleton_roots))
        }
        singleton_replicas = {
            root: whole_value_replicas.get(root, 1) for root in singleton_roots
        }
        if singleton_roots:
            singleton_epoch_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_singleton_epochs",
                numel=(
                    f"{len(singleton_roots)} * {epoch_replicas} * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                dtype=torch.uint32,
                zero_init=True,
            )

        whole_value_producer_roots = sorted(
            {
                producer
                for producer, _ in whole_value_edges
                if producer not in singleton_roots
            }
        )
        whole_value_indices = {
            root: index for index, root in enumerate(whole_value_producer_roots)
        }
        whole_value_counter_arg: str | None = None
        if whole_value_producer_roots:
            whole_value_counter_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_whole_value_arrivals",
                numel=(
                    f"{len(whole_value_producer_roots)} * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                dtype=torch.uint32,
                zero_init=True,
            )
        whole_value_epoch_roots = sorted(
            root
            for root in whole_value_producer_roots
            if whole_value_replicas[root] > 1
        )
        whole_value_epoch_indices = {
            root: index for index, root in enumerate(whole_value_epoch_roots)
        }
        whole_value_epoch_arg: str | None = None
        if whole_value_epoch_roots:
            whole_value_epoch_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_whole_value_epochs",
                numel=(
                    f"{len(whole_value_epoch_roots)} * {epoch_replicas} * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                dtype=torch.uint32,
                zero_init=True,
            )

        reduction_fanout_indices = {
            root: index for index, root in enumerate(reduction_fanout_plans)
        }
        reduction_fanout_arrival_arg: str | None = None
        if reduction_fanout_plans:
            reduction_fanout_arrival_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_reduction_fanout_arrivals",
                numel=(
                    f"{len(reduction_fanout_plans)} * {_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                dtype=torch.uint32,
                zero_init=True,
            )
        reduction_fanout_partition_offsets: dict[int, int] = {}
        reduction_fanout_partition_count = 0
        for root, plan in reduction_fanout_plans.items():
            if plan.upstream_root is None:
                continue
            reduction_fanout_partition_offsets[root] = reduction_fanout_partition_count
            reduction_fanout_partition_count += plan.task_count
        reduction_fanout_partition_arg: str | None = None
        if reduction_fanout_partition_count:
            reduction_fanout_partition_arg = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_reduction_fanout_partitions",
                numel=str(reduction_fanout_partition_count),
                dtype=torch.uint32,
                zero_init=True,
            )

        partitioned_pipeline_args: dict[str, str] = {}
        partitioned_pipeline_offsets: dict[int, dict[str, int]] = {}
        partitioned_pipeline_plan_indices = {
            root: index for index, root in enumerate(partitioned_pipeline_plans)
        }
        if partitioned_pipeline_plans:
            totals = {
                "primary_arrivals": 0,
                "first_side_arrivals": 0,
                "second_side_arrivals": 0,
                "primary_ready": 0,
                "materialized_ready": 0,
                "partition_map_arrivals": 0,
                "reduction_arrivals": 0,
            }
            for root, plan in partitioned_pipeline_plans.items():
                partitioned_pipeline_offsets[root] = dict(totals)
                totals["primary_arrivals"] += (
                    plan.group_count
                    * plan.primary_members_per_partition
                    // plan.finalize_partition_block
                )
                totals["first_side_arrivals"] += (
                    plan.group_count + plan.finalize_partition_block - 1
                ) // plan.finalize_partition_block
                totals["second_side_arrivals"] += plan.group_count
                totals["primary_ready"] += plan.group_count
                totals["materialized_ready"] += plan.group_count
                totals["partition_map_arrivals"] += plan.partition_map_counter_count
                if plan.final_reduce_root is not None:
                    totals["reduction_arrivals"] += plan.final_reduce_tasks
            for name, count in totals.items():
                argument = self._register_tile_dependency_state(
                    device_function,
                    name_hint=f"tile_dependency_{name}",
                    numel=str(max(1, count)),
                    dtype=torch.uint32,
                    zero_init=True,
                )
                partitioned_pipeline_args[name] = argument
            pipeline_ready = self._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_partitioned_pipeline_ready",
                numel=(
                    f"{len(partitioned_pipeline_plans)} * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                dtype=torch.uint32,
                zero_init=True,
            )
            partitioned_pipeline_args["pipeline_ready"] = pipeline_ready
            if epoch_replicas > 1 and any(
                plan.downstream_tasks > _TILE_DEPENDENCY_DIRECT_POLL_FANOUT_LIMIT
                for plan in partitioned_pipeline_plans.values()
            ):
                partitioned_pipeline_args["pipeline_epochs"] = (
                    self._register_tile_dependency_state(
                        device_function,
                        name_hint="tile_dependency_partitioned_pipeline_epochs",
                        numel=(
                            f"{len(partitioned_pipeline_plans)} * "
                            f"{epoch_replicas} * "
                            f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                        ),
                        dtype=torch.uint32,
                        zero_init=True,
                    )
                )

        stage_root_ranges = self._tile_dependency_stage_root_ranges()
        result: list[ast.stmt] = [
            statement_from_string(f"{epoch_var} = tl.load({epoch_arg} + {worker}) + 1")
        ]
        start_expr = "0"
        consumed_partitioned_roots = {
            root
            for plan in partitioned_pipeline_plans.values()
            for root in plan.continuation_roots
        }
        consumed_reduction_fanout_roots = {
            root
            for plan in reduction_fanout_plans.values()
            for root in plan.continuation_roots
        }
        consumed_task_continuation_roots = {
            plan.consumer_root for plan in task_continuation_plans
        } | {plan.cohort.consumer_root for plan in task_continuation_pipeline_plans}

        def singleton_dependency(root: int) -> tuple[str, str] | None:
            if root not in singleton_roots:
                return None
            assert singleton_epoch_arg is not None
            plan_index = singleton_indices[root]
            replicas = singleton_replicas[root]
            return (
                (
                    f"{singleton_epoch_arg} + "
                    f"({plan_index} * {epoch_replicas} + "
                    f"(({worker}) % {replicas})) * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                f"tl.cast({epoch_var}, tl.uint32)",
            )

        whole_value_incoming: dict[int, tuple[int, ...]] = {
            consumer: tuple(
                sorted(
                    producer
                    for producer, target in whole_value_edges
                    if target == consumer
                )
            )
            for consumer in {consumer for _, consumer in whole_value_edges}
        }

        def root_completion_dependency(root: int) -> tuple[str, str]:
            singleton = singleton_dependency(root)
            if singleton is not None:
                return singleton
            assert static_task_counts is not None
            replicas = whole_value_replicas[root]
            if replicas > 1:
                assert whole_value_epoch_arg is not None
                epoch_index = whole_value_epoch_indices[root]
                return (
                    (
                        f"{whole_value_epoch_arg} + "
                        f"({epoch_index * epoch_replicas} + "
                        f"(({worker}) % {replicas})) * "
                        f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                    ),
                    f"tl.cast({epoch_var}, tl.uint32)",
                )
            assert whole_value_counter_arg is not None
            active_workers = active_worker_count(root)
            return (
                (
                    f"{whole_value_counter_arg} + "
                    f"{whole_value_indices[root] * _TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                f"tl.cast({epoch_var}, tl.uint32) * tl.cast({active_workers}, tl.uint32)",
            )

        def whole_value_input_dependencies(
            root: int,
        ) -> tuple[tuple[str, str], ...]:
            producers = whole_value_incoming.get(root, ())
            return tuple(root_completion_dependency(producer) for producer in producers)

        def whole_value_input_dependency(root: int) -> tuple[str, str] | None:
            dependencies = whole_value_input_dependencies(root)
            if not dependencies:
                return None
            if len(dependencies) != 1:
                raise exc.TileDependencyScheduleError(
                    "this migration-only structural plan supports one "
                    f"whole-value predecessor, got {len(dependencies)} for root {root}"
                )
            return dependencies[0]

        def root_completion_publication(
            root: int, active_workers: int
        ) -> list[ast.stmt]:
            if root not in whole_value_indices:
                return []
            assert whole_value_counter_arg is not None
            completion_counter = (
                f"{whole_value_counter_arg} + "
                f"{whole_value_indices[root] * _TILE_DEPENDENCY_COUNTER_STRIDE}"
            )
            result = [self._tile_dependency_publication_barrier(device_function)]
            replicas = whole_value_replicas[root]
            if replicas == 1:
                result.append(
                    statement_from_string(
                        f"tl.atomic_add({completion_counter}, 1, "
                        "sem='release', scope='gpu')"
                    )
                )
                return result
            assert whole_value_epoch_arg is not None
            completion_previous = device_function.new_var(
                "tile_dependency_whole_value_previous", dce=False
            )
            epoch_index = whole_value_epoch_indices[root]
            epoch_base = (
                f"{whole_value_epoch_arg} + "
                f"{epoch_index * epoch_replicas * _TILE_DEPENDENCY_COUNTER_STRIDE}"
            )
            result.extend(
                [
                    statement_from_string(
                        f"{completion_previous} = "
                        f"tl.atomic_add({completion_counter}, 1, "
                        "sem='acq_rel', scope='gpu')"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"({completion_previous} % "
                            f"tl.cast({active_workers}, tl.uint32)) == "
                            f"tl.cast({active_workers - 1}, tl.uint32)"
                        ),
                        body=self._publish_tile_dependency_epoch(
                            device_function,
                            base=epoch_base,
                            epoch=epoch_var,
                            replicas=replicas,
                        ),
                        orelse=[],
                    ),
                ]
            )
            return result

        root_axis_order: list[tuple[int, ...]] = []
        root_axis_counts: list[dict[int, int]] = []
        all_axis_counts: dict[int, int] = {}
        block_sizes: dict[int, int] = {}
        for root in range(len(self.cases)):
            geometry = self._static_case_geometry(root, device_function)
            assert geometry is not None
            axis_order, root_counts, axis_block_sizes = geometry
            root_axis_order.append(axis_order)
            root_axis_counts.append(root_counts)
            all_axis_counts.update(root_counts)
            block_sizes.update(axis_block_sizes)

        for waits in generic_task_waits.values():
            for wait in waits:
                assert wait.predecessor_map is not None
                for axis in wait.predecessor_map.axes:
                    if axis.consumer_block_id in block_sizes:
                        continue
                    geometry = self._static_block_axis_geometry(
                        axis.consumer_block_id, device_function
                    )
                    assert geometry is not None
                    count, block = geometry
                    all_axis_counts[axis.consumer_block_id] = count
                    block_sizes[axis.consumer_block_id] = block
        for plan in access_cohort_plans:
            for axis in plan.axes:
                if axis.consumer_block_id in block_sizes:
                    continue
                geometry = self._static_block_axis_geometry(
                    axis.consumer_block_id, device_function
                )
                assert geometry is not None
                count, block = geometry
                all_axis_counts[axis.consumer_block_id] = count
                block_sizes[axis.consumer_block_id] = block

        access_cohorts_by_producer: dict[int, list[ForEachProgramID._AccessCohort]] = {}
        access_cohorts_by_consumer: dict[int, list[ForEachProgramID._AccessCohort]] = {}
        for plan in access_cohort_plans:
            access_cohorts_by_producer.setdefault(plan.producer_root, []).append(plan)
            access_cohorts_by_consumer.setdefault(plan.consumer_root, []).append(plan)
        task_continuation_by_producer = {
            plan.producer_root: plan for plan in task_continuation_plans
        }
        task_continuation_pipeline_by_producer = {
            plan.continuation.producer_root: plan
            for plan in task_continuation_pipeline_plans
        }

        def task_coordinates(
            task: str,
            axis_order: tuple[int, ...],
            counts: dict[int, int],
        ) -> dict[int, str]:
            coordinates: dict[int, str] = {}
            multiplier = 1
            for block_id in axis_order:
                coordinates[block_id] = (
                    f"((({task}) // {multiplier}) % {counts[block_id]})"
                )
                multiplier *= counts[block_id]
            return coordinates

        def cohort_counter(
            plan: ForEachProgramID._AccessCohort,
            *,
            producer_coordinates: dict[int, str],
            milestone: int | str,
        ) -> str:
            assert access_cohort_arg is not None
            producer_counts = root_axis_counts[plan.producer_root]
            outer_terms: list[str] = []
            multiplier = 1
            for block_id in plan.outer_producer_axes:
                outer_terms.append(f"({producer_coordinates[block_id]}) * {multiplier}")
                multiplier *= producer_counts[block_id]
            outer_key = " + ".join(outer_terms) or "0"
            stride = 1 if plan.is_per_coordinate else _TILE_DEPENDENCY_COUNTER_STRIDE
            return (
                f"{access_cohort_arg} + {access_cohort_offsets[plan]} + "
                f"(({outer_key}) * {plan.milestone_count} + ({milestone})) * "
                f"{stride}"
            )

        def cohort_publication(
            plan: ForEachProgramID._AccessCohort,
            local_task: str,
        ) -> list[ast.stmt]:
            coordinates = task_coordinates(
                local_task,
                root_axis_order[plan.producer_root],
                root_axis_counts[plan.producer_root],
            )
            stream_coordinate = coordinates[plan.producer_stream_axis]

            if plan.is_per_coordinate:
                return [
                    statement_from_string(
                        f"tl.atomic_add({cohort_counter(plan, producer_coordinates=coordinates, milestone=stream_coordinate)}, "
                        "1, sem='release', scope='gpu')"
                    )
                ]

            def publish(stage: int) -> ast.stmt:
                return statement_from_string(
                    f"tl.atomic_add({cohort_counter(plan, producer_coordinates=coordinates, milestone=stage)}, "
                    "1, sem='release', scope='gpu')"
                )

            result: list[ast.stmt] = [publish(len(plan.stage_sizes) - 1)]
            stage_end = sum(plan.stage_sizes)
            for stage in range(len(plan.stage_sizes) - 2, -1, -1):
                stage_end -= plan.stage_sizes[stage + 1]
                result = [
                    create(
                        ast.If,
                        test=expr_from_string(f"({stream_coordinate}) < {stage_end}"),
                        body=[publish(stage)],
                        orelse=result,
                    )
                ]
            return result

        def cohort_consumer_body(
            plan: ForEachProgramID._AccessCohort,
            local_task: str,
            logical_pid: str,
            extra_argument_names: tuple[str, ...],
        ) -> ast.stmt:
            consumer_coordinates = task_coordinates(
                local_task,
                root_axis_order[plan.consumer_root],
                root_axis_counts[plan.consumer_root],
            )
            axes_by_producer = {axis.producer_block_id: axis for axis in plan.axes}
            producer_coordinates = {
                block_id: consumer_coordinates[
                    axes_by_producer[block_id].consumer_block_id
                ]
                for block_id in plan.outer_producer_axes
            }
            if plan.is_per_coordinate:
                assert plan.consumer_stream_coordinate is not None
                scheduled_root_body = (
                    _clone_opaque_statements_with_access_iteration_wait(
                        case_bodies[plan.consumer_root],
                        access_ids=frozenset(plan.access_ids),
                        loop_id=plan.consumer_loop_id,
                        wait=self._wait_for_counter(
                            device_function=device_function,
                            counter=cohort_counter(
                                plan,
                                producer_coordinates=producer_coordinates,
                                milestone=plan.consumer_stream_coordinate,
                            ),
                            target=f"tl.cast({epoch_var}, tl.uint32)",
                            prefix="tile_dependency_cohort_wait",
                        ),
                    )
                )
                return self._outline_tile_dependency_region(
                    device_function,
                    name_hint=f"tile_dependency_root_{plan.consumer_root}",
                    body=[
                        statement_from_string(f"{self.shared_pid_var} = {logical_pid}"),
                        *scheduled_root_body,
                    ],
                    extra_argument_names=extra_argument_names,
                )
            stage_waits = tuple(
                tuple(
                    self._wait_for_counter(
                        device_function=device_function,
                        counter=cohort_counter(
                            plan,
                            producer_coordinates=producer_coordinates,
                            milestone=stage,
                        ),
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({stage_size}, tl.uint32)"
                        ),
                        prefix="tile_dependency_cohort_wait",
                    )
                )
                for stage, stage_size in enumerate(plan.stage_sizes)
            )
            stream_block = block_sizes[plan.consumer_stream_axis]
            split_offsets = tuple(
                str(offset * stream_block) for offset in plan.stage_offsets[1:-1]
            )
            scheduled_root_body = _clone_opaque_statements_with_access_stages(
                case_bodies[plan.consumer_root],
                access_ids=frozenset(plan.access_ids),
                split_offsets=split_offsets,
                stage_waits=stage_waits,
            )
            return self._outline_tile_dependency_region(
                device_function,
                name_hint=f"tile_dependency_root_{plan.consumer_root}",
                body=[
                    statement_from_string(f"{self.shared_pid_var} = {logical_pid}"),
                    *scheduled_root_body,
                ],
                extra_argument_names=extra_argument_names,
            )

        def task_scheduled_body(
            root: int,
            local_task: str,
            logical_pid: str,
            extra_argument_names: tuple[str, ...],
            *,
            force_noinline: bool = False,
        ) -> list[ast.stmt]:
            body: list[ast.stmt] = []
            has_task_scheduling = root in access_cohorts_by_consumer
            continuation_plan = task_continuation_by_producer.get(root)
            scheduled_local_task = local_task
            scheduled_logical_pid = logical_pid
            continuation_consumer_task: str | None = None
            if continuation_plan is not None:
                has_task_scheduling = True
                continuation_consumer_task = device_function.new_var(
                    "tile_dependency_continuation_task", dce=True
                )
                continuation_local = device_function.new_var(
                    "tile_dependency_continuation_local", dce=True
                )
                physical_task = device_function.new_var(
                    "tile_dependency_continuation_physical_task", dce=True
                )
                partition = continuation_plan.partition
                consumer_coordinates = task_coordinates(
                    continuation_consumer_task,
                    partition.consumer_axis_order,
                    root_axis_counts[continuation_plan.consumer_root],
                )
                producer_coordinates = {
                    axis.producer_block_id: (
                        f"({consumer_coordinates[axis.consumer_block_id]}) * "
                        f"{axis.scale} + {axis.offset}"
                    )
                    for axis in partition.outer_axes
                }
                partition_consumer_coordinate = consumer_coordinates[
                    partition.partition_consumer_block_id
                ]
                segment_offsets: list[int] = []
                running_segment_offset = 0
                for segment in partition.segments:
                    segment_offsets.append(running_segment_offset)
                    running_segment_offset += segment.length
                partition_coordinate = ""
                for segment, segment_offset in reversed(
                    tuple(zip(partition.segments, segment_offsets, strict=True))
                ):
                    segment_coordinate = (
                        f"{segment.begin} + "
                        f"({partition_consumer_coordinate}) * "
                        f"{partition.partition_consumer_stride} + "
                        f"({continuation_local}) - {segment_offset}"
                    )
                    if not partition_coordinate:
                        partition_coordinate = segment_coordinate
                    else:
                        partition_coordinate = (
                            f"tl.where(({continuation_local}) < "
                            f"{segment_offset + segment.length}, "
                            f"{segment_coordinate}, {partition_coordinate})"
                        )
                assert partition_coordinate
                producer_coordinates[partition.partition_producer_block_id] = (
                    partition_coordinate
                )
                physical_task_terms: list[str] = []
                physical_task_multiplier = 1
                for block_id in partition.producer_axis_order:
                    physical_task_terms.append(
                        f"({producer_coordinates[block_id]}) * "
                        f"{physical_task_multiplier}"
                    )
                    physical_task_multiplier *= root_axis_counts[root][block_id]
                physical_task_expr = " + ".join(physical_task_terms)
                body.extend(
                    [
                        statement_from_string(
                            f"{continuation_consumer_task} = "
                            f"({local_task}) // {continuation_plan.fanin}"
                        ),
                        statement_from_string(
                            f"{continuation_local} = "
                            f"({local_task}) % {continuation_plan.fanin}"
                        ),
                        statement_from_string(
                            f"{physical_task} = {physical_task_expr}"
                        ),
                    ]
                )
                scheduled_local_task = physical_task
                scheduled_logical_pid = f"{case_offsets[root]} + {physical_task}"
            emitted_access_waits: set[tuple[int, tuple[object, ...]]] = set()
            access_split_offsets: dict[int, str] = {}
            if root in generic_task_waits:
                has_task_scheduling = True
                assert dependency_plan is not None and task_event_arg is not None
                for wait in generic_task_waits[root]:
                    producer_root = dependency_plan.event(wait.event_id).producer_root
                    if wait.placement == "root_entry":
                        body.extend(
                            self._emit_root_admission_task_wait(
                                wait=wait,
                                device_function=device_function,
                                task_event_arg=task_event_arg,
                                task_event_offset=task_event_offsets[producer_root],
                                epoch_var=epoch_var,
                                consumer_task=scheduled_local_task,
                                consumer_axis_order=root_axis_order[root],
                                consumer_axis_counts=root_axis_counts[root],
                                producer_axis_order=root_axis_order[producer_root],
                                producer_axis_counts=root_axis_counts[producer_root],
                                block_sizes=block_sizes,
                            )
                        )
                        continue
                    assert wait.predecessor_map is not None
                    wait_key = (wait.event_id, wait.predecessor_map.axes)
                    if wait_key in emitted_access_waits:
                        continue
                    emitted_access_waits.add(wait_key)
                    nested_axes = {
                        axis.consumer_block_id
                        for axis in wait.predecessor_map.axes
                        if axis.consumer_block_id not in root_axis_counts[root]
                    }
                    if len(nested_axes) == 1:
                        nested_axis = nested_axes.pop()
                        nested_count = all_axis_counts[nested_axis]
                        if nested_count > 1:
                            access_id = wait.consumer_access_id
                            assert access_id is not None
                            access_split_offsets[access_id] = str(
                                (nested_count // 2) * block_sizes[nested_axis]
                            )
                    body.extend(
                        self._emit_access_preflight_task_wait(
                            wait=wait,
                            device_function=device_function,
                            task_event_arg=task_event_arg,
                            task_event_offset=task_event_offsets[producer_root],
                            epoch_var=epoch_var,
                            consumer_task=scheduled_local_task,
                            consumer_axis_order=root_axis_order[root],
                            consumer_axis_counts=all_axis_counts,
                            producer_axis_order=root_axis_order[producer_root],
                            producer_axis_counts=root_axis_counts[producer_root],
                            block_sizes=block_sizes,
                        )
                    )
            if cohort_plans := access_cohorts_by_consumer.get(root):
                if len(cohort_plans) != 1:
                    raise AssertionError(
                        "coarsened access scheduling requires one event per "
                        "consumer root"
                    )
                opaque_call = cohort_consumer_body(
                    cohort_plans[0],
                    scheduled_local_task,
                    scheduled_logical_pid,
                    extra_argument_names,
                )
            elif access_split_offsets:
                scheduled_root_body = _clone_opaque_statements_with_split_access_loops(
                    case_bodies[root], access_split_offsets
                )
                opaque_call = self._outline_tile_dependency_region(
                    device_function,
                    name_hint=f"tile_dependency_root_{root}",
                    body=[
                        statement_from_string(
                            f"{self.shared_pid_var} = {scheduled_logical_pid}"
                        ),
                        *scheduled_root_body,
                    ],
                    extra_argument_names=extra_argument_names,
                )
            else:
                opaque_call = self._outline_opaque_tile_body(
                    device_function,
                    root=root,
                    logical_pid=scheduled_logical_pid,
                    body=case_bodies[root],
                    extra_argument_names=extra_argument_names,
                    noinline=force_noinline,
                )
            body.append(opaque_call)
            publications: list[ast.stmt] = []
            if root in task_event_offsets:
                assert task_event_arg is not None
                publications.append(
                    statement_from_string(
                        f"tl.atomic_xchg({task_event_arg} + "
                        f"{task_event_offsets[root]} + ({scheduled_local_task}), "
                        f"{epoch_var}, sem='release', scope='gpu')"
                    )
                )
            for plan in access_cohorts_by_producer.get(root, ()):
                publications.extend(cohort_publication(plan, scheduled_local_task))
            if publications or continuation_plan is not None:
                has_task_scheduling = True
                body.append(self._tile_dependency_publication_barrier(device_function))
                body.extend(publications)
            if continuation_plan is not None:
                assert task_continuation_arg is not None
                assert continuation_consumer_task is not None
                previous = device_function.new_var(
                    "tile_dependency_continuation_previous", dce=False
                )
                arrival_counter = (
                    f"{task_continuation_arg} + "
                    f"{task_continuation_offsets[continuation_plan]} + "
                    f"{continuation_consumer_task}"
                )
                consumer_call = self._outline_opaque_tile_body(
                    device_function,
                    root=continuation_plan.consumer_root,
                    logical_pid=(
                        f"{case_offsets[continuation_plan.consumer_root]} + "
                        f"{continuation_consumer_task}"
                    ),
                    body=case_bodies[continuation_plan.consumer_root],
                    extra_argument_names=(continuation_consumer_task,),
                )
                consumer_publications: list[ast.stmt] = []
                if continuation_plan.consumer_root in task_event_offsets:
                    assert task_event_arg is not None
                    consumer_publications.append(
                        statement_from_string(
                            f"tl.atomic_xchg({task_event_arg} + "
                            f"{task_event_offsets[continuation_plan.consumer_root]} + "
                            f"{continuation_consumer_task}, {epoch_var}, "
                            "sem='release', scope='gpu')"
                        )
                    )
                for plan in access_cohorts_by_producer.get(
                    continuation_plan.consumer_root, ()
                ):
                    consumer_publications.extend(
                        cohort_publication(plan, continuation_consumer_task)
                    )
                last_arrival_body = [consumer_call]
                if consumer_publications:
                    last_arrival_body.append(
                        self._tile_dependency_publication_barrier(device_function)
                    )
                    last_arrival_body.extend(consumer_publications)
                body.extend(
                    [
                        statement_from_string(
                            f"{previous} = tl.atomic_add({arrival_counter}, 1, "
                            "sem='acq_rel', scope='gpu')"
                        ),
                        create(
                            ast.If,
                            test=expr_from_string(
                                f"{previous} == "
                                f"tl.cast({epoch_var}, tl.uint32) * "
                                f"tl.cast({continuation_plan.fanin}, tl.uint32) - 1"
                            ),
                            body=last_arrival_body,
                            orelse=[],
                        ),
                    ]
                )
            if not has_task_scheduling:
                return body
            return [
                self._outline_tile_dependency_region(
                    device_function,
                    name_hint=f"tile_dependency_root_{root}_scheduled_task",
                    body=body,
                    extra_argument_names=extra_argument_names,
                    noinline=True,
                )
            ]

        def task_continuation_pipeline_body(
            plan: ForEachProgramID._TaskContinuationPipeline,
        ) -> list[ast.stmt]:
            producer_root = plan.continuation.producer_root
            consumer_root = plan.cohort.consumer_root
            producer_task = device_function.new_var(
                "tile_dependency_pipeline_producer_task", dce=True
            )
            consumer_task = device_function.new_var(
                "tile_dependency_pipeline_consumer_task", dce=True
            )
            producer_body = task_scheduled_body(
                producer_root,
                producer_task,
                f"{case_offsets[producer_root]} + {producer_task}",
                (producer_task, epoch_var),
            )
            consumer_body = task_scheduled_body(
                consumer_root,
                consumer_task,
                f"{case_offsets[consumer_root]} + {consumer_task}",
                (consumer_task, epoch_var),
            )
            body: list[ast.stmt] = []
            if (
                input_dependency := whole_value_input_dependency(producer_root)
            ) is not None:
                body.extend(
                    self._wait_for_counter(
                        device_function=device_function,
                        counter=input_dependency[0],
                        target=input_dependency[1],
                        prefix="tile_dependency_pipeline_input_wait",
                    )
                )
            body.extend(
                [
                    statement_from_string(f"{producer_task} = {worker}"),
                    *_clone_opaque_statements(producer_body),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"({worker}) < {plan.tail_producer_tasks}"
                        ),
                        body=[
                            statement_from_string(
                                f"{producer_task} = {plan.worker_count} + {worker}"
                            ),
                            *_clone_opaque_statements(producer_body),
                        ],
                        orelse=[],
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"({worker}) >= {plan.consumer_worker_begin}"
                        ),
                        body=[
                            statement_from_string(
                                f"{consumer_task} = "
                                f"{worker} - {plan.consumer_worker_begin}"
                            ),
                            *consumer_body,
                        ],
                        orelse=[],
                    ),
                ]
            )
            return [
                create(
                    ast.If,
                    test=expr_from_string(f"({worker}) < {plan.worker_count}"),
                    body=body,
                    orelse=[],
                )
            ]

        for _stage_index, (boundary, root_range) in enumerate(
            zip(boundaries, stage_root_ranges, strict=True)
        ):
            root_begin, root_end = root_range
            if root_end != root_begin + 1:
                structural_roots = (
                    set(partitioned_pipeline_plans)
                    | set(reduction_fanout_plans)
                    | consumed_partitioned_roots
                    | consumed_reduction_fanout_roots
                    | consumed_task_continuation_roots
                    | set(task_continuation_pipeline_by_producer)
                )
                if structural_roots.intersection(range(root_begin, root_end)):
                    raise exc.TileDependencyScheduleError(
                        "a tile-dependency structural plan cannot share a stage "
                        "with an independent root yet"
                    )
                assert static_task_counts is not None
                for stage_root in range(root_begin, root_end):
                    if stage_root in singleton_roots and any(
                        edge.producer_root == stage_root
                        for edge in dependency_plan.edges
                    ):
                        assert singleton_epoch_arg is not None
                        singleton_task_body = task_scheduled_body(
                            stage_root,
                            "0",
                            str(case_offsets[stage_root]),
                            (epoch_var,),
                            force_noinline=True,
                        )
                        result.extend(
                            self._emit_opaque_singleton_root(
                                task_body=singleton_task_body,
                                device_function=device_function,
                                worker=worker,
                                epoch_var=epoch_var,
                                ready_arg=singleton_epoch_arg,
                                ready_index=singleton_indices[stage_root],
                                epoch_replicas=singleton_replicas[stage_root],
                                ready_replica_stride=epoch_replicas,
                                input_dependencies=whole_value_input_dependencies(
                                    stage_root
                                ),
                            )
                        )
                        continue
                    stage_workers = min(
                        static_task_counts[stage_root], launch_worker_limit
                    )
                    stage_body: list[ast.stmt] = []
                    for producer_root in whole_value_incoming.get(stage_root, ()):
                        counter, target = root_completion_dependency(producer_root)
                        stage_body.extend(
                            self._wait_for_counter(
                                device_function=device_function,
                                counter=counter,
                                target=target,
                                prefix="tile_dependency_whole_value_wait",
                            )
                        )
                    stage_body.append(
                        create(
                            ast.For,
                            target=create(
                                ast.Name,
                                id=strategy.virtual_pid_var,
                                ctx=ast.Store(),
                            ),
                            iter=expr_from_string(
                                f"tl.range(({worker}) + ({case_offsets[stage_root]}), "
                                f"({case_offsets[stage_root] + static_task_counts[stage_root]}), "
                                f"{strategy.grid_size_expr})"
                            ),
                            body=task_scheduled_body(
                                stage_root,
                                (
                                    f"({strategy.virtual_pid_var}) - "
                                    f"{case_offsets[stage_root]}"
                                ),
                                strategy.virtual_pid_var,
                                (strategy.virtual_pid_var,),
                            ),
                            orelse=[],
                            type_comment=None,
                        )
                    )
                    stage_body.extend(
                        root_completion_publication(stage_root, stage_workers)
                    )
                    result.append(
                        create(
                            ast.If,
                            test=expr_from_string(f"({worker}) < {stage_workers}"),
                            body=stage_body,
                            orelse=[],
                        )
                    )
                start_expr = boundary
                continue
            task_continuation_pipeline = (
                task_continuation_pipeline_by_producer.get(root_begin)
                if root_end == root_begin + 1
                else None
            )
            if task_continuation_pipeline is not None:
                result.extend(
                    task_continuation_pipeline_body(task_continuation_pipeline)
                )
                start_expr = boundary
                continue
            if root_end == root_begin + 1 and root_begin in (
                consumed_partitioned_roots
                | consumed_reduction_fanout_roots
                | consumed_task_continuation_roots
            ):
                start_expr = boundary
                continue
            reduction_fanout_plan = (
                reduction_fanout_plans.get(root_begin)
                if root_end == root_begin + 1
                else None
            )
            if reduction_fanout_plan is not None:
                assert reduction_fanout_arrival_arg is not None
                fanout_task = device_function.new_var(
                    "tile_dependency_reduction_fanout_task", dce=True
                )
                fanout_counter = (
                    f"{reduction_fanout_arrival_arg} + "
                    f"{reduction_fanout_indices[root_begin] * _TILE_DEPENDENCY_COUNTER_STRIDE}"
                )
                producer_call = self._outline_opaque_tile_body(
                    device_function,
                    root=reduction_fanout_plan.producer_root,
                    logical_pid=(
                        f"{case_offsets[reduction_fanout_plan.producer_root]} + "
                        f"{fanout_task}"
                    ),
                    body=case_bodies[reduction_fanout_plan.producer_root],
                    extra_argument_names=(fanout_task,),
                )
                producer_prefix: list[ast.stmt] = []
                producer_input = whole_value_input_dependency(root_begin)
                if producer_input is not None:
                    producer_prefix.extend(
                        self._wait_for_counter(
                            device_function=device_function,
                            counter=producer_input[0],
                            target=producer_input[1],
                            prefix="tile_dependency_reduction_fanout_input_wait",
                        )
                    )
                producer_body: list[ast.stmt]
                producer_loop_task_count: int
                if reduction_fanout_plan.upstream_root is None:
                    producer_loop_task_count = reduction_fanout_plan.task_count
                    producer_body = [
                        *producer_prefix,
                        producer_call,
                        self._tile_dependency_publication_barrier(device_function),
                        statement_from_string(
                            f"tl.atomic_add({fanout_counter}, 1, "
                            "sem='release', scope='gpu')"
                        ),
                    ]
                else:
                    assert reduction_fanout_partition_arg is not None
                    partition_offset = reduction_fanout_partition_offsets[root_begin]
                    fanout_partition = device_function.new_var(
                        "tile_dependency_reduction_fanout_partition", dce=True
                    )
                    partition_counter = (
                        f"{reduction_fanout_partition_arg} + "
                        f"{partition_offset} + {fanout_partition}"
                    )
                    upstream_call = self._outline_opaque_tile_body(
                        device_function,
                        root=reduction_fanout_plan.upstream_root,
                        logical_pid=(
                            f"{case_offsets[reduction_fanout_plan.upstream_root]} + "
                            f"{fanout_task}"
                        ),
                        body=case_bodies[reduction_fanout_plan.upstream_root],
                        extra_argument_names=(fanout_task,),
                    )
                    producer_loop_task_count = reduction_fanout_plan.upstream_tasks
                    producer_body = [
                        *producer_prefix,
                        upstream_call,
                        self._tile_dependency_publication_barrier(device_function),
                        statement_from_string(
                            f"{fanout_partition} = {fanout_task} // "
                            f"{reduction_fanout_plan.upstream_tasks_per_partition}"
                        ),
                        statement_from_string(
                            f"tl.atomic_add({partition_counter}, 1, "
                            "sem='release', scope='gpu')"
                        ),
                    ]
                    producer_call = self._outline_opaque_tile_body(
                        device_function,
                        root=reduction_fanout_plan.producer_root,
                        logical_pid=(
                            f"{case_offsets[reduction_fanout_plan.producer_root]} + "
                            f"{fanout_task}"
                        ),
                        body=case_bodies[reduction_fanout_plan.producer_root],
                        name_suffix="partition_consumer",
                        extra_argument_names=(fanout_task,),
                    )
                    producer_call = create(
                        ast.If,
                        test=expr_from_string(
                            f"{worker} < {reduction_fanout_plan.task_count}"
                        ),
                        body=[
                            statement_from_string(f"{fanout_task} = {worker}"),
                            *self._wait_for_counter(
                                device_function=device_function,
                                counter=(
                                    f"{reduction_fanout_partition_arg} + "
                                    f"{partition_offset} + {fanout_task}"
                                ),
                                target=(
                                    f"tl.cast({epoch_var}, tl.uint32) * "
                                    f"tl.cast("
                                    f"{reduction_fanout_plan.upstream_tasks_per_partition}, "
                                    "tl.uint32)"
                                ),
                                prefix="tile_dependency_partitioned_input_wait",
                            ),
                            producer_call,
                            self._tile_dependency_publication_barrier(device_function),
                            statement_from_string(
                                f"tl.atomic_add({fanout_counter}, 1, "
                                "sem='release', scope='gpu')"
                            ),
                        ],
                        orelse=[],
                    )
                reduction_call = self._outline_opaque_tile_body(
                    device_function,
                    root=reduction_fanout_plan.reduction_root,
                    logical_pid=str(case_offsets[reduction_fanout_plan.reduction_root]),
                    body=case_bodies[reduction_fanout_plan.reduction_root],
                    name_suffix="replicated",
                    extra_argument_names=(fanout_task,),
                )
                consumer_call = self._outline_opaque_tile_body(
                    device_function,
                    root=reduction_fanout_plan.consumer_root,
                    logical_pid=(
                        f"{case_offsets[reduction_fanout_plan.consumer_root]} + "
                        f"{fanout_task}"
                    ),
                    body=case_bodies[reduction_fanout_plan.consumer_root],
                    extra_argument_names=(fanout_task,),
                )
                consumer_body = [
                    *self._wait_for_counter(
                        device_function=device_function,
                        counter=fanout_counter,
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({reduction_fanout_plan.task_count}, tl.uint32)"
                        ),
                        prefix="tile_dependency_reduction_fanout_wait",
                    ),
                    reduction_call,
                    self._tile_dependency_publication_barrier(device_function),
                    consumer_call,
                    *root_completion_publication(
                        reduction_fanout_plan.consumer_root,
                        reduction_fanout_plan.task_count,
                    ),
                ]
                result.extend(
                    [
                        create(
                            ast.For,
                            target=create(ast.Name, id=fanout_task, ctx=ast.Store()),
                            iter=expr_from_string(
                                f"tl.range({worker}, "
                                f"{producer_loop_task_count}, "
                                f"{strategy.grid_size_expr})"
                            ),
                            body=producer_body,
                            orelse=[],
                            type_comment=None,
                        ),
                        *(
                            [producer_call]
                            if reduction_fanout_plan.upstream_root is not None
                            else []
                        ),
                        create(
                            ast.If,
                            test=expr_from_string(
                                f"{worker} < {reduction_fanout_plan.task_count}"
                            ),
                            body=[
                                statement_from_string(f"{fanout_task} = {worker}"),
                                *consumer_body,
                            ],
                            orelse=[],
                        ),
                    ]
                )
                start_expr = boundary
                continue
            partitioned_pipeline_plan = (
                partitioned_pipeline_plans.get(root_begin)
                if root_end == root_begin + 1
                else None
            )
            if partitioned_pipeline_plan is not None:
                assert static_task_counts is not None
                offsets = partitioned_pipeline_offsets[root_begin]
                input_dependency = whole_value_input_dependency(root_begin)
                result.extend(
                    self._emit_partitioned_dependency_pipeline(
                        partitioned_pipeline_plan,
                        case_bodies=case_bodies,
                        case_offsets=case_offsets,
                        strategy=strategy,
                        device_function=device_function,
                        worker=worker,
                        epoch_var=epoch_var,
                        primary_arrivals=f"{partitioned_pipeline_args['primary_arrivals']} + {offsets['primary_arrivals']}",
                        first_side_arrivals=f"{partitioned_pipeline_args['first_side_arrivals']} + {offsets['first_side_arrivals']}",
                        second_side_arrivals=f"{partitioned_pipeline_args['second_side_arrivals']} + {offsets['second_side_arrivals']}",
                        primary_ready=f"{partitioned_pipeline_args['primary_ready']} + {offsets['primary_ready']}",
                        materialized_ready=f"{partitioned_pipeline_args['materialized_ready']} + {offsets['materialized_ready']}",
                        partition_map_arrivals=f"{partitioned_pipeline_args['partition_map_arrivals']} + {offsets['partition_map_arrivals']}",
                        reduction_arrivals=f"{partitioned_pipeline_args['reduction_arrivals']} + {offsets['reduction_arrivals']}",
                        pipeline_ready=partitioned_pipeline_args["pipeline_ready"],
                        pipeline_epochs=partitioned_pipeline_args.get(
                            "pipeline_epochs"
                        ),
                        plan_index=partitioned_pipeline_plan_indices[root_begin],
                        input_ready_counter=(
                            input_dependency[0]
                            if input_dependency is not None
                            else None
                        ),
                        input_ready_target=(
                            input_dependency[1]
                            if input_dependency is not None
                            else None
                        ),
                    )
                )
                start_expr = boundary
                continue
            if root_end == root_begin + 1 and root_begin in singleton_roots:
                assert singleton_epoch_arg is not None
                singleton_task_body = task_scheduled_body(
                    root_begin,
                    "0",
                    str(case_offsets[root_begin]),
                    (epoch_var,),
                    force_noinline=True,
                )
                result.extend(
                    self._emit_opaque_singleton_root(
                        task_body=singleton_task_body,
                        device_function=device_function,
                        worker=worker,
                        epoch_var=epoch_var,
                        ready_arg=singleton_epoch_arg,
                        ready_index=singleton_indices[root_begin],
                        epoch_replicas=singleton_replicas[root_begin],
                        ready_replica_stride=epoch_replicas,
                        input_dependencies=whole_value_input_dependencies(root_begin),
                    )
                )
                start_expr = boundary
                continue
            task_body = task_scheduled_body(
                root_begin,
                f"({strategy.virtual_pid_var}) - {case_offsets[root_begin]}",
                strategy.virtual_pid_var,
                (strategy.virtual_pid_var,),
            )
            task_loop = create(
                ast.For,
                target=create(ast.Name, id=strategy.virtual_pid_var, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.range(({worker}) + ({start_expr}), ({boundary}), {strategy.grid_size_expr})"
                ),
                body=task_body,
                orelse=[],
                type_comment=None,
            )
            incoming_roots = whole_value_incoming.get(root_begin, ())
            publishes_completion = root_begin in whole_value_indices
            if not incoming_roots and not publishes_completion:
                result.append(task_loop)
                start_expr = boundary
                continue

            assert static_task_counts is not None
            active_workers = active_worker_count(root_begin)
            active_body: list[ast.stmt] = []
            for producer_root in incoming_roots:
                counter, target = root_completion_dependency(producer_root)
                active_body.extend(
                    self._wait_for_counter(
                        device_function=device_function,
                        counter=counter,
                        target=target,
                        prefix="tile_dependency_whole_value_wait",
                    )
                )
            active_body.append(task_loop)
            if publishes_completion:
                active_body.extend(
                    root_completion_publication(root_begin, active_workers)
                )
            result.append(
                create(
                    ast.If,
                    test=expr_from_string(f"({worker}) < {active_workers}"),
                    body=active_body,
                    orelse=[],
                )
            )
            start_expr = boundary
        result.append(
            statement_from_string(f"tl.store({epoch_arg} + {worker}, {epoch_var})")
        )
        if (
            tuple(_ast_fingerprint(body) for body in case_bodies)
            != opaque_case_fingerprints
        ):
            raise AssertionError(
                "tile-dependency lowering mutated an opaque source tile body"
            )
        return result

    @staticmethod
    def _validate_tile_dependency_plan_coverage(
        *,
        singleton_roots: frozenset[int],
        partitioned_pipeline_plans: dict[
            int, ForEachProgramID._PartitionedDependencyPipeline
        ],
        reduction_fanout_plans: dict[int, ForEachProgramID._OneWaveReductionFanout],
        task_ready_edges: frozenset[tuple[int, int]],
        whole_value_edges: frozenset[tuple[int, int]],
    ) -> None:
        """Verify that every dependence has a synchronization path.

        Structural plans replace whole-value synchronization where the access
        relation proves a finer schedule. Remaining explicit tensor edges use
        the stage counter already emitted by the generic lowering. This is a
        first-class whole-value dependency, not permission to reinterpret an
        opaque computation body.
        """
        covered: set[tuple[int, int]] = set()
        for plan in partitioned_pipeline_plans.values():
            covered.update(plan.covered_edges)
        for plan in reduction_fanout_plans.values():
            covered.update(plan.covered_edges)
        covered.update(task_ready_edges)
        covered.update(whole_value_edges)

        # An opaque singleton root publishes one completion epoch consumed by
        # the next recognized producer chain.  The root body itself is never
        # split or rewritten.
        chain_starts = set(partitioned_pipeline_plans)
        for singleton_root in singleton_roots:
            if singleton_root + 1 in chain_starts:
                covered.add((singleton_root, singleton_root + 1))

        def is_ordered(producer: int, consumer: int) -> bool:
            pending = [producer]
            visited: set[int] = set()
            while pending:
                current = pending.pop()
                if current == consumer:
                    return True
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(
                    target for source, target in covered if source == current
                )
            return False

        dependency_plan = HostFunction.current().device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        for dependency in dependency_plan.edges:
            if not is_ordered(dependency.producer_root, dependency.consumer_root):
                raise exc.TileDependencyScheduleError(
                    f"{dependency.producer_root}->{dependency.consumer_root} "
                    f"through allocations {sorted(dependency.tensor_names)!r} has no "
                    "TileDependency synchronization path"
                )

        # The remaining partitioned-attention emitter still uses prior-value
        # modulo election. Unlike the generic continuation's epoch-relative
        # exact target, that legacy protocol requires power-of-two fan-ins.
        fanins: list[int] = []
        for plan in partitioned_pipeline_plans.values():
            fanins.extend(
                (
                    plan.finalize_partition_block * plan.tiles_per_member,
                    plan.tiles_per_member,
                )
            )
        invalid_fanin = next(
            (fanin for fanin in fanins if fanin <= 0 or fanin & (fanin - 1)), None
        )
        if invalid_fanin is not None:
            raise exc.TileDependencyScheduleError(
                "32-bit last-arrival counters require a power-of-two fan-in; "
                f"got {invalid_fanin}"
            )

    def _tile_dependency_stage_root_ranges(self) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        begin = 0
        for index in range(1, len(self.case_phases) + 1):
            if (
                index == len(self.case_phases)
                or self.case_phases[index] != self.case_phases[index - 1]
            ):
                result.append((begin, index))
                begin = index
        return result

    def _extract_case_bodies(self, base_body: list[ast.stmt]) -> list[list[ast.stmt]]:
        if len(self.cases) == 1:
            return [base_body]
        assert len(base_body) >= 2
        node = base_body[1]
        result: list[list[ast.stmt]] = []
        while isinstance(node, ast.If):
            result.append(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
                continue
            result.append(node.orelse)
            break
        assert len(result) == len(self.cases)
        return result

    _AccessCohort = AccessCohortPlan
    _TaskContinuation = TaskContinuationPlan
    _TaskContinuationPipeline = TaskContinuationPipelinePlan

    @dataclasses.dataclass(frozen=True)
    class _OneWaveReductionFanout:
        """Replicate one pure scalar reduction across its one-wave consumers."""

        producer_root: int
        reduction_root: int
        consumer_root: int
        task_count: int
        upstream_root: int | None = None
        upstream_tasks: int = 0
        upstream_tasks_per_partition: int = 0

        @property
        def start_root(self) -> int:
            return (
                self.producer_root if self.upstream_root is None else self.upstream_root
            )

        @property
        def covered_edges(self) -> tuple[tuple[int, int], ...]:
            edges = (
                (self.producer_root, self.reduction_root),
                (self.producer_root, self.consumer_root),
                (self.reduction_root, self.consumer_root),
            )
            if self.upstream_root is None:
                return edges
            return ((self.upstream_root, self.producer_root), *edges)

        @property
        def continuation_roots(self) -> frozenset[int]:
            return frozenset(
                (self.producer_root, self.reduction_root, self.consumer_root)
            ) - {self.start_root}

    @dataclasses.dataclass(frozen=True)
    class _PartitionedDependencyPipeline:
        producer_root: int
        partition_finalize_root: int
        materialize_root: int
        partition_map_root: int
        partition_reduce_root: int
        final_reduce_root: int | None
        map_root: int
        downstream_root: int
        producer_tasks: int
        group_count: int
        primary_members_per_partition: int
        finalize_partition_block: int
        tiles_per_member: int
        materialize_tasks_per_partition: int
        partition_tasks: int
        partition_tasks_per_partition: int
        reduction_tasks: int
        reduction_tasks_per_partition: int
        final_reduce_tasks: int
        final_reduce_tasks_per_partition: int
        subpartitions_per_partition: int
        output_tasks_per_partition: int
        downstream_tasks: int
        consumer_affinity_order: bool

        @property
        def covered_edges(self) -> tuple[tuple[int, int], ...]:
            prefix = (
                (self.producer_root, self.partition_finalize_root),
                (self.partition_finalize_root, self.materialize_root),
                (self.partition_finalize_root, self.partition_map_root),
                (self.materialize_root, self.partition_map_root),
                (self.partition_map_root, self.partition_reduce_root),
            )
            if self.final_reduce_root is None:
                return (
                    *prefix,
                    (self.partition_reduce_root, self.map_root),
                    (self.map_root, self.downstream_root),
                )
            return (
                *prefix,
                (self.partition_reduce_root, self.final_reduce_root),
                (self.final_reduce_root, self.map_root),
                (self.map_root, self.downstream_root),
            )

        @property
        def continuation_roots(self) -> frozenset[int]:
            return frozenset(
                root
                for root in (
                    self.partition_finalize_root,
                    self.materialize_root,
                    self.partition_map_root,
                    self.partition_reduce_root,
                    self.final_reduce_root,
                    self.map_root,
                )
                if root is not None
            )

        @property
        def partition_map_counter_count(self) -> int:
            return self.group_count * self.subpartitions_per_partition

        @property
        def completion_tasks(self) -> int:
            return (
                self.reduction_tasks
                if self.final_reduce_root is None
                else self.final_reduce_tasks
            )

    @staticmethod
    def _match_opaque_singleton_roots(
        task_counts: list[int],
    ) -> frozenset[int]:
        """Select one-task roots without inspecting or rewriting their bodies."""
        dependency_plan = HostFunction.current().device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        participating_roots = {
            root
            for edge in dependency_plan.edges
            for root in (edge.producer_root, edge.consumer_root)
        }
        return frozenset(
            root
            for root, task_count in enumerate(task_counts)
            if task_count == 1 and root in participating_roots
        )

    @staticmethod
    def _case_pid_info(case: ProgramIDs) -> list[PIDInfo]:
        from .tile_strategy import L2GroupingProgramIDs

        if isinstance(case, L2GroupingProgramIDs):
            assert case.parent_strategy is not None
            return case.parent_strategy.pid_info
        return case.pid_info

    def _instantiated_task_families(
        self,
        device_function: DeviceFunction,
    ) -> tuple[InstantiatedTaskFamily, ...] | None:
        """Bind DeviceIR logical task families to one physical configuration."""
        from .tile_strategy import L2GroupingProgramIDs

        logical_families = HostFunction.current().device_ir.task_families
        if len(logical_families) != len(self.cases):
            return None
        result: list[InstantiatedTaskFamily] = []
        for root, logical_family in enumerate(logical_families):
            geometry = self._static_case_geometry(root, device_function)
            if geometry is None:
                return None
            physical_axis_order, axis_counts, block_sizes = geometry
            if set(logical_family.logical_axis_order) != set(physical_axis_order):
                return None
            result.append(
                InstantiatedTaskFamily(
                    root=root,
                    logical_axis_order=logical_family.logical_axis_order,
                    physical_axis_order=physical_axis_order,
                    axis_counts_items=tuple(
                        (block_id, axis_counts[block_id])
                        for block_id in physical_axis_order
                    ),
                    block_sizes_items=tuple(
                        (block_id, block_sizes[block_id])
                        for block_id in physical_axis_order
                    ),
                    has_nontrivial_pid_remap=isinstance(
                        self.cases[root], L2GroupingProgramIDs
                    ),
                )
            )
        return tuple(result)

    def _static_case_task_counts(
        self, device_function: DeviceFunction
    ) -> list[int] | None:
        result: list[int] = []
        for root in range(len(self.cases)):
            geometry = self._static_case_geometry(root, device_function)
            if geometry is None:
                return None
            _, axis_counts, _ = geometry
            result.append(math.prod(axis_counts.values()))
        return result

    def _static_case_geometry(
        self,
        root: int,
        device_function: DeviceFunction,
    ) -> tuple[tuple[int, ...], dict[int, int], dict[int, int]] | None:
        axes = self._static_case_axes(root, device_function)
        if axes is None:
            return None
        infos = self._case_pid_info(self.cases[root])
        axis_order = tuple(info.block_id for info in infos)
        axis_counts = {
            info.block_id: (numel + block - 1) // block
            for info, (numel, block) in zip(infos, axes, strict=True)
        }
        block_sizes = {
            info.block_id: block for info, (_, block) in zip(infos, axes, strict=True)
        }
        return axis_order, axis_counts, block_sizes

    @staticmethod
    def _static_block_axis_geometry(
        block_id: int,
        device_function: DeviceFunction,
    ) -> tuple[int, int] | None:
        """Return ``(task_count, block_size)`` for one statically sized axis."""
        env = CompileEnvironment.current()
        try:
            numel_expr = env.block_sizes[block_id].numel
            if not numel_expr.is_number:
                return None
            numel = int(numel_expr)
            block = int(
                env.block_sizes[block_id].from_config_assert(device_function.config)
            )
        except (KeyError, TypeError, ValueError):
            return None
        return (numel + block - 1) // block, block

    def _match_one_wave_reduction_fanouts(
        self,
        singleton_roots: frozenset[int],
        case_bodies: list[list[ast.stmt]],
        device_function: DeviceFunction,
    ) -> dict[int, _OneWaveReductionFanout]:
        """Find a pure singleton reduction feeding one resident tile wave.

        The reduction body is not rewritten.  Each consumer worker executes
        the same opaque singleton body immediately before its own opaque tile.
        This removes a global scalar publication hop while preserving the
        reduction's statement and iteration order.  The match is deliberately
        conservative: producer and consumer domains must be identical, the
        producer must fit in one resident wave, and the singleton may write
        only storage consumed by that immediate consumer.
        """
        task_counts = self._static_case_task_counts(device_function)
        if task_counts is None:
            return {}
        device_ir = HostFunction.current().device_ir
        dependency_plan = device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        resident_workers = CompileEnvironment.current().config_spec.num_sm * cast(
            "int", device_function.config.get("num_sm_multiplier", 1)
        )
        result: dict[int, ForEachProgramID._OneWaveReductionFanout] = {}
        for reduction_root in singleton_roots:
            producer_root = reduction_root - 1
            consumer_root = reduction_root + 1
            if producer_root < 0 or consumer_root >= len(self.cases):
                continue
            task_count = task_counts[producer_root]
            if (
                task_count <= 1
                or task_count != task_counts[consumer_root]
                or task_count > resident_workers
            ):
                continue
            if not (
                dependency_plan.edges_between(producer_root, reduction_root)
                and dependency_plan.edges_between(reduction_root, consumer_root)
                and dependency_plan.edges_between(producer_root, consumer_root)
            ):
                continue
            reduction_writes = {
                access.allocation_id
                for access in dependency_plan.accesses
                if access.root == reduction_root
                and access.kind == "store"
                and access.allocation_id >= 0
            }
            outgoing_allocations = {
                edge.allocation_id
                for edge in dependency_plan.edges
                if edge.producer_root == reduction_root
                and edge.consumer_root == consumer_root
                and edge.allocation_id >= 0
            }
            if not reduction_writes or reduction_writes != outgoing_allocations:
                continue
            if any(
                edge.consumer_root != consumer_root
                for edge in dependency_plan.edges
                if edge.producer_root == reduction_root
            ):
                continue
            if any(
                isinstance(node, ast.Attribute) and node.attr.startswith("atomic_")
                for statement in case_bodies[reduction_root]
                for node in ast.walk(statement)
            ):
                continue
            upstream_root: int | None = None
            upstream_tasks = 0
            upstream_tasks_per_partition = 0
            candidate_upstream = producer_root - 1
            if candidate_upstream >= 0:
                upstream_edges = dependency_plan.edges_between(
                    candidate_upstream, producer_root
                )
                upstream_allocations = {
                    edge.allocation_id
                    for edge in upstream_edges
                    if edge.allocation_id >= 0
                }
                if len(upstream_allocations) == 1:
                    candidate_tasks = task_counts[candidate_upstream]
                    candidate_axes = self._static_case_axes(
                        candidate_upstream, device_function
                    )
                    producer_axes = self._static_case_axes(
                        producer_root, device_function
                    )
                    candidate_nontrivial = [
                        (numel, block)
                        for numel, block in candidate_axes or ()
                        if (numel + block - 1) // block > 1
                    ]
                    producer_nontrivial = [
                        (numel, block)
                        for numel, block in producer_axes or ()
                        if (numel + block - 1) // block > 1
                    ]
                    if (
                        candidate_tasks % task_count == 0
                        and len(candidate_nontrivial) == 1
                        and len(producer_nontrivial) == 1
                    ):
                        tasks_per_partition = candidate_tasks // task_count
                        candidate_numel, candidate_block = candidate_nontrivial[0]
                        partition_numel, partition_block = producer_nontrivial[0]
                        vector_extent = candidate_block * tasks_per_partition
                        full_vector_axes = {
                            numel
                            for numel, block in producer_axes or ()
                            if numel == block and numel > 1
                        }
                        allocation_id = upstream_allocations.pop()
                        upstream_write_count = sum(
                            access.allocation_id == allocation_id
                            and access.kind == "store"
                            and access.root == candidate_upstream
                            for access in dependency_plan.accesses
                        )
                        producer_read_count = sum(
                            access.allocation_id == allocation_id
                            and access.kind == "load"
                            and access.root == producer_root
                            for access in dependency_plan.accesses
                        )
                        if (
                            partition_block == 1
                            and candidate_numel == partition_numel * vector_extent
                            and vector_extent in full_vector_axes
                            and upstream_write_count == 1
                            and producer_read_count == 1
                        ):
                            upstream_root = candidate_upstream
                            upstream_tasks = candidate_tasks
                            upstream_tasks_per_partition = tasks_per_partition
            plan = self._OneWaveReductionFanout(
                producer_root=producer_root,
                reduction_root=reduction_root,
                consumer_root=consumer_root,
                task_count=task_count,
                upstream_root=upstream_root,
                upstream_tasks=upstream_tasks,
                upstream_tasks_per_partition=upstream_tasks_per_partition,
            )
            result[plan.start_root] = plan
        return result

    def _static_case_axes(
        self, root: int, device_function: DeviceFunction
    ) -> list[tuple[int, int]] | None:
        env = CompileEnvironment.current()
        task_families = HostFunction.current().device_ir.task_families
        task_family = task_families[root] if root < len(task_families) else None
        result: list[tuple[int, int]] = []
        for info in self._case_pid_info(self.cases[root]):
            logical_axis = (
                task_family.axis(info.block_id) if task_family is not None else None
            )
            numel_expr = logical_axis.extent if logical_axis is not None else info.numel
            if isinstance(numel_expr, str) or numel_expr is None:
                return None
            if isinstance(numel_expr, int):
                numel = numel_expr
            elif isinstance(numel_expr, torch.SymInt):
                numel = int(env.size_hint(numel_expr))
            elif getattr(numel_expr, "is_number", False):
                numel = int(numel_expr)
            else:
                return None
            try:
                block = int(
                    env.block_sizes[info.block_id].from_config_assert(
                        device_function.config
                    )
                )
            except (KeyError, TypeError, ValueError):
                return None
            result.append((numel, block))
        return result

    def _match_partitioned_dependency_pipeline(
        self,
        case_bodies: list[list[ast.stmt]],
        device_function: DeviceFunction,
    ) -> dict[int, _PartitionedDependencyPipeline]:
        """Match a static partitioned producer/consumer pipeline.

        This is a topology match over adjacent opaque tile programs with
        statically compatible task domains.  The reduction tail may contain
        either one level or two nested levels.  Absolute sizes, source names,
        and computation operators are irrelevant. A failed proof is rejected
        by the coverage check rather than triggering a computation rewrite.
        """
        task_counts = self._static_case_task_counts(device_function)
        if task_counts is None:
            return {}
        dependency_plan = HostFunction.current().device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        result: dict[int, ForEachProgramID._PartitionedDependencyPipeline] = {}
        for producer_root in range(len(self.cases) - 6):
            partition_finalize_root = producer_root + 1
            materialize_root = producer_root + 2
            partition_map_root = producer_root + 3
            partition_reduce_root = producer_root + 4
            if not (
                len(
                    dependency_plan.edges_between(
                        producer_root, partition_finalize_root
                    )
                )
                >= 1
                and len(
                    dependency_plan.edges_between(
                        partition_finalize_root, materialize_root
                    )
                )
                >= 1
                and len(
                    dependency_plan.edges_between(
                        partition_finalize_root, partition_map_root
                    )
                )
                >= 1
                and len(
                    dependency_plan.edges_between(materialize_root, partition_map_root)
                )
                >= 1
                and len(
                    dependency_plan.edges_between(
                        partition_map_root, partition_reduce_root
                    )
                )
                == 2
            ):
                continue
            final_reduce_root: int | None = None
            map_root = producer_root + 5
            downstream_root = producer_root + 6
            if not (
                len(dependency_plan.edges_between(partition_reduce_root, map_root)) == 1
                and len(dependency_plan.edges_between(map_root, downstream_root)) == 2
            ):
                if producer_root + 7 >= len(self.cases):
                    continue
                final_reduce_root = producer_root + 5
                map_root = producer_root + 6
                downstream_root = producer_root + 7
                if not (
                    len(
                        dependency_plan.edges_between(
                            partition_reduce_root, final_reduce_root
                        )
                    )
                    == 2
                    and len(dependency_plan.edges_between(final_reduce_root, map_root))
                    == 1
                    and len(dependency_plan.edges_between(map_root, downstream_root))
                    == 2
                ):
                    continue
            finalize_axes = self._static_case_axes(
                partition_finalize_root, device_function
            )
            materialize_axes = self._static_case_axes(materialize_root, device_function)
            if finalize_axes is None or materialize_axes is None:
                continue
            finalize_nontrivial = [
                (numel, block)
                for numel, block in finalize_axes
                if (numel + block - 1) // block > 1
            ]
            materialize_nontrivial = [
                (numel, block)
                for numel, block in materialize_axes
                if (numel + block - 1) // block > 1
            ]
            if len(finalize_nontrivial) != 1 or len(materialize_nontrivial) != 1:
                continue
            finalized_members, finalize_partition_block = finalize_nontrivial[0]
            partition_extent, partition_block = materialize_nontrivial[0]
            if partition_block != 1:
                continue
            producer_tasks = task_counts[producer_root]
            materialize_tasks = task_counts[materialize_root]
            if materialize_tasks != partition_extent:
                continue
            geometry = _partitioned_materialization_geometry(
                producer_tasks=producer_tasks,
                finalized_members=finalized_members,
                finalize_partition_block=finalize_partition_block,
                materialize_tasks=materialize_tasks,
            )
            if geometry is None:
                continue
            (
                group_count,
                primary_members_per_partition,
                tiles_per_member,
                materialize_tasks_per_partition,
            ) = geometry
            producer_axes = self._static_case_axes(producer_root, device_function)
            if producer_axes is None:
                continue
            producer_nontrivial = [
                (numel + block - 1) // block
                for numel, block in producer_axes
                if (numel + block - 1) // block > 1
            ]
            if producer_nontrivial != [producer_tasks]:
                continue
            partition_tasks = task_counts[partition_map_root]
            if partition_tasks % group_count:
                continue
            reduction_tasks = task_counts[partition_reduce_root]
            if reduction_tasks % group_count:
                continue
            final_reduce_tasks = (
                reduction_tasks
                if final_reduce_root is None
                else task_counts[final_reduce_root]
            )
            if final_reduce_tasks % group_count:
                continue
            output_map_tasks = task_counts[map_root]
            if output_map_tasks % group_count:
                continue
            reduction_tasks_per_partition = reduction_tasks // group_count
            final_reduce_tasks_per_partition = final_reduce_tasks // group_count
            output_tasks_per_partition = output_map_tasks // group_count
            if output_tasks_per_partition % final_reduce_tasks_per_partition:
                continue
            if reduction_tasks_per_partition % final_reduce_tasks_per_partition:
                continue
            subpartitions_per_partition = (
                reduction_tasks_per_partition // final_reduce_tasks_per_partition
            )
            if partition_tasks // group_count % subpartitions_per_partition:
                continue

            # Prove the mixed-radix task maps used by lowering from physical
            # PID-axis order.  The partition map must vary partition first;
            # for a two-level reduction, the first reduction must vary the
            # nested subpartition first and retain the final task as its outer
            # coordinate.  This is a scheduling-layout proof only: no tile
            # body or tensor expression is inspected or changed.
            partition_axis_counts = [
                (numel + block - 1) // block
                for numel, block in self._static_case_axes(
                    partition_map_root, device_function
                )
                or ()
                if (numel + block - 1) // block > 1
            ]
            if not partition_axis_counts or partition_axis_counts[0] != group_count:
                continue
            if math.prod(partition_axis_counts[1:]) != partition_tasks // group_count:
                continue
            if final_reduce_root is not None:
                reduction_axis_counts = [
                    (numel + block - 1) // block
                    for numel, block in self._static_case_axes(
                        partition_reduce_root, device_function
                    )
                    or ()
                    if (numel + block - 1) // block > 1
                ]
                if (
                    not reduction_axis_counts
                    or reduction_axis_counts[0] != subpartitions_per_partition
                    or math.prod(reduction_axis_counts[1:]) != final_reduce_tasks
                ):
                    continue
            resident_workers = CompileEnvironment.current().config_spec.num_sm * cast(
                "int", device_function.config.get("num_sm_multiplier", 1)
            )
            tasks_per_group = (primary_members_per_partition + 2) * tiles_per_member
            affinity_prefix = min(group_count, resident_workers // tasks_per_group)
            physical_completed_members = resident_workers // tiles_per_member
            physical_prefix = min(
                group_count,
                max(
                    0,
                    physical_completed_members
                    - group_count * (primary_members_per_partition + 1),
                ),
            )
            schedule = HostFunction.current().device_ir.tile_dependency_schedule
            assert schedule is not None and schedule.policy is not None
            requested_order = schedule.policy.producer_order
            consumer_affinity_order = (
                affinity_prefix > physical_prefix
                if requested_order is None
                else requested_order == "consumer_major"
            )
            result[producer_root] = self._PartitionedDependencyPipeline(
                producer_root=producer_root,
                partition_finalize_root=partition_finalize_root,
                materialize_root=materialize_root,
                partition_map_root=partition_map_root,
                partition_reduce_root=partition_reduce_root,
                final_reduce_root=final_reduce_root,
                map_root=map_root,
                downstream_root=downstream_root,
                producer_tasks=producer_tasks,
                group_count=group_count,
                primary_members_per_partition=primary_members_per_partition,
                finalize_partition_block=finalize_partition_block,
                tiles_per_member=tiles_per_member,
                materialize_tasks_per_partition=materialize_tasks_per_partition,
                partition_tasks=partition_tasks,
                partition_tasks_per_partition=partition_tasks // group_count,
                reduction_tasks=reduction_tasks,
                reduction_tasks_per_partition=reduction_tasks_per_partition,
                final_reduce_tasks=final_reduce_tasks,
                final_reduce_tasks_per_partition=final_reduce_tasks_per_partition,
                subpartitions_per_partition=subpartitions_per_partition,
                output_tasks_per_partition=output_tasks_per_partition,
                downstream_tasks=task_counts[downstream_root],
                consumer_affinity_order=consumer_affinity_order,
            )
        return result

    @staticmethod
    def _emit_root_admission_task_wait(
        *,
        wait: WaitSpec,
        device_function: DeviceFunction,
        task_event_arg: str,
        task_event_offset: int,
        epoch_var: str,
        consumer_task: str,
        consumer_axis_order: tuple[int, ...],
        consumer_axis_counts: dict[int, int],
        producer_axis_order: tuple[int, ...],
        producer_axis_counts: dict[int, int],
        block_sizes: dict[int, int],
    ) -> list[ast.stmt]:
        """Wait for the exact producer tasks required by one consumer task."""
        consumer_coordinates: dict[int, str] = {}
        multiplier = 1
        for block_id in consumer_axis_order:
            count = consumer_axis_counts[block_id]
            consumer_coordinates[block_id] = (
                f"((({consumer_task}) // {multiplier}) % {count})"
            )
            multiplier *= count

        return ForEachProgramID._emit_task_wait(
            wait=wait,
            device_function=device_function,
            task_event_arg=task_event_arg,
            task_event_offset=task_event_offset,
            epoch_var=epoch_var,
            consumer_coordinates=consumer_coordinates,
            consumer_axis_counts=consumer_axis_counts,
            producer_axis_order=producer_axis_order,
            producer_axis_counts=producer_axis_counts,
            block_sizes=block_sizes,
        )

    @staticmethod
    def _emit_access_preflight_task_wait(
        *,
        wait: WaitSpec,
        device_function: DeviceFunction,
        task_event_arg: str,
        task_event_offset: int,
        epoch_var: str,
        consumer_task: str,
        consumer_axis_order: tuple[int, ...],
        consumer_axis_counts: dict[int, int],
        producer_axis_order: tuple[int, ...],
        producer_axis_counts: dict[int, int],
        block_sizes: dict[int, int],
    ) -> list[ast.stmt]:
        """Wait for every nested coordinate used by one opaque consumer task."""
        predecessor_map = wait.predecessor_map
        assert predecessor_map is not None

        consumer_coordinates: dict[int, str] = {}
        multiplier = 1
        for block_id in consumer_axis_order:
            count = consumer_axis_counts[block_id]
            consumer_coordinates[block_id] = (
                f"((({consumer_task}) // {multiplier}) % {count})"
            )
            multiplier *= count

        nested_coordinates: list[tuple[str, int]] = []
        for axis in predecessor_map.axes:
            block_id = axis.consumer_block_id
            if block_id in consumer_coordinates:
                continue
            coordinate = device_function.new_var(
                "tile_dependency_consumer_coordinate", dce=True
            )
            consumer_coordinates[block_id] = coordinate
            nested_coordinates.append((coordinate, consumer_axis_counts[block_id]))

        body: list[ast.stmt] = ForEachProgramID._emit_task_wait(
            wait=wait,
            device_function=device_function,
            task_event_arg=task_event_arg,
            task_event_offset=task_event_offset,
            epoch_var=epoch_var,
            consumer_coordinates=consumer_coordinates,
            consumer_axis_counts=consumer_axis_counts,
            producer_axis_order=producer_axis_order,
            producer_axis_counts=producer_axis_counts,
            block_sizes=block_sizes,
        )
        for coordinate, count in reversed(nested_coordinates):
            loop = create(
                ast.For,
                target=create(ast.Name, id=coordinate, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.range(0, {count}, num_stages=1, "
                    "disallow_acc_multi_buffer=True)"
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            body = [loop]
        return body

    @staticmethod
    def _emit_task_wait(
        *,
        wait: WaitSpec,
        device_function: DeviceFunction,
        task_event_arg: str,
        task_event_offset: int,
        epoch_var: str,
        consumer_coordinates: dict[int, str],
        consumer_axis_counts: dict[int, int],
        producer_axis_order: tuple[int, ...],
        producer_axis_counts: dict[int, int],
        block_sizes: dict[int, int],
    ) -> list[ast.stmt]:
        """Wait for exact producer tasks from explicit consumer coordinates."""
        predecessor_map = wait.predecessor_map
        assert predecessor_map is not None

        predecessor_coordinates: dict[int, str] = {}
        predecessor_loops: list[tuple[str, str, str]] = []
        for axis in predecessor_map.axes:
            consumer_coordinate = consumer_coordinates[axis.consumer_block_id]
            consumer_block = (
                1 if axis.consumer_is_scalar else block_sizes[axis.consumer_block_id]
            )
            producer_block = (
                1 if axis.producer_is_scalar else block_sizes[axis.producer_block_id]
            )
            producer_count = producer_axis_counts[axis.producer_block_id]
            if (
                consumer_block == producer_block
                and axis.consumer_offset == axis.producer_offset
                and consumer_axis_counts.get(axis.consumer_block_id) == producer_count
            ):
                predecessor_coordinates[axis.producer_block_id] = consumer_coordinate
                continue

            begin = (
                f"(({consumer_coordinate}) * {consumer_block} + {axis.consumer_offset})"
            )
            end = f"(({begin}) + {consumer_block - 1})"
            first = (
                f"tl.maximum(0, (({begin}) - {axis.producer_offset}) // "
                f"{producer_block})"
            )
            last = (
                f"tl.minimum({producer_count - 1}, "
                f"(({end}) - {axis.producer_offset}) // {producer_block})"
            )
            coordinate = device_function.new_var(
                "tile_dependency_predecessor", dce=True
            )
            predecessor_coordinates[axis.producer_block_id] = coordinate
            predecessor_loops.append((coordinate, first, last))

        producer_task_terms: list[str] = []
        multiplier = 1
        for block_id in producer_axis_order:
            coordinate = predecessor_coordinates[block_id]
            producer_task_terms.append(f"({coordinate}) * {multiplier}")
            multiplier *= producer_axis_counts[block_id]
        producer_task = " + ".join(producer_task_terms) or "0"
        body: list[ast.stmt] = ForEachProgramID._wait_for_counter(
            device_function=device_function,
            counter=(f"{task_event_arg} + {task_event_offset} + ({producer_task})"),
            target=f"tl.cast({epoch_var}, tl.uint32)",
            prefix="tile_dependency_task_wait",
        )
        for coordinate, first, last in reversed(predecessor_loops):
            body = [
                create(
                    ast.For,
                    target=create(ast.Name, id=coordinate, ctx=ast.Store()),
                    iter=expr_from_string(f"tl.range({first}, ({last}) + 1)"),
                    body=body,
                    orelse=[],
                    type_comment=None,
                )
            ]
        return body

    @staticmethod
    def _wait_for_counter(
        *,
        device_function: DeviceFunction,
        counter: str,
        target: str,
        prefix: str,
    ) -> list[ast.stmt]:
        value = device_function.new_var(prefix, dce=False)
        sync = device_function.new_var(f"{prefix}_sync", dce=False)
        load = (
            "tl.inline_asm_elementwise("
            "asm='ld.acquire.gpu.global.u32 $0, [$1];', "
            "constraints='=r,l', "
            f"args=[{counter}], dtype=tl.uint32, is_pure=False, pack=1)"
        )
        return [
            statement_from_string(f"{value} = {load}"),
            create(
                ast.While,
                test=expr_from_string(f"{value} != ({target})"),
                body=[statement_from_string(f"{value} = {load}")],
                orelse=[],
            ),
            statement_from_string(
                f"{sync} = tl.inline_asm_elementwise("
                "asm='bar.warp.sync 0xffffffff; mov.u32 $0, $1;', "
                "constraints='=r,r', args=[tl.arange(0, 32)], "
                "dtype=tl.uint32, is_pure=False, pack=1)"
            ),
        ]

    @staticmethod
    def _tile_dependency_publication_barrier(
        device_function: DeviceFunction,
    ) -> ast.stmt:
        if cast("int", device_function.config.get("num_warps", 1)) != 1:
            return statement_from_string("tl.debug_barrier()")
        sync = device_function.new_var("tile_dependency_publication_sync", dce=False)
        return statement_from_string(
            f"{sync} = tl.inline_asm_elementwise("
            "asm='bar.warp.sync 0xffffffff; mov.u32 $0, $1;', "
            "constraints='=r,r', args=[tl.arange(0, 32)], "
            "dtype=tl.uint32, is_pure=False, pack=1)"
        )

    @staticmethod
    def _publish_tile_dependency_epoch(
        device_function: DeviceFunction,
        *,
        base: str,
        epoch: str,
        replicas: int,
    ) -> list[ast.stmt]:
        """Release-publish one epoch, vectorizing replicated cache lines."""
        if replicas == 1:
            return [
                statement_from_string(
                    f"tl.atomic_xchg({base}, {epoch}, sem='release', scope='gpu')"
                )
            ]
        offsets = device_function.new_var("tile_dependency_epoch_offsets", dce=True)
        return [
            statement_from_string(
                f"{offsets} = tl.arange(0, {replicas}) * "
                f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
            ),
            statement_from_string(
                f"tl.atomic_xchg({base} + {offsets}, {epoch}, "
                "sem='release', scope='gpu')"
            ),
        ]

    @staticmethod
    def _register_tile_dependency_state(
        device_function: DeviceFunction,
        *,
        name_hint: str,
        numel: str,
        dtype: torch.dtype,
        zero_init: bool,
    ) -> str:
        """Register launch-persistent global state owned by the Triton launcher."""
        like = next(
            argument
            for argument in device_function.arguments
            if isinstance(argument, TensorArg) and argument._host_str is not None
        )
        name = device_function.new_var(name_hint, dce=False)
        device_function.wrapper_only_params.append(name)
        device_function.triton_persistent_state_args.append(name)
        device_function.triton_persistent_state_specs.append(
            (like.host_str(), numel, str(dtype), zero_init)
        )
        return name

    @staticmethod
    def _outline_tile_dependency_region(
        device_function: DeviceFunction,
        *,
        name_hint: str,
        body: list[ast.stmt],
        extra_argument_names: tuple[str, ...] = (),
        noinline: bool = False,
    ) -> ast.stmt:
        """Outline a scheduled region while keeping its computation opaque."""
        helper_name, arguments = device_function.register_triton_noinline_helper(
            name_hint,
            body,
            extra_argument_names=extra_argument_names,
            noinline=noinline,
        )
        return statement_from_string(f"{helper_name}({', '.join(arguments)})")

    def _outline_opaque_tile_body(
        self,
        device_function: DeviceFunction,
        *,
        root: int,
        logical_pid: str,
        body: list[ast.stmt],
        name_suffix: str = "",
        extra_argument_names: tuple[str, ...] = (),
        noinline: bool = False,
    ) -> ast.stmt:
        """Create a noinline call containing exactly one original tile body."""
        suffix = f"_{name_suffix}" if name_suffix else ""
        return self._outline_tile_dependency_region(
            device_function,
            name_hint=f"tile_dependency_root_{root}{suffix}",
            body=[
                statement_from_string(f"{self.shared_pid_var} = {logical_pid}"),
                *_clone_opaque_statements(body),
            ],
            extra_argument_names=extra_argument_names,
            noinline=noinline,
        )

    def _emit_opaque_singleton_root(
        self,
        *,
        task_body: list[ast.stmt],
        device_function: DeviceFunction,
        worker: str,
        epoch_var: str,
        ready_arg: str,
        ready_index: int,
        epoch_replicas: int,
        ready_replica_stride: int | None = None,
        input_dependencies: tuple[tuple[str, str], ...] = (),
    ) -> list[ast.stmt]:
        """Run one original logical tile and publish its completion epoch.

        ``task_body`` comes from the same generic task scheduler used by every
        other root.  The singleton wrapper adds only root-level admission and
        completion publication.
        """
        replica_stride = ready_replica_stride or epoch_replicas
        ready_base = (
            f"{ready_arg} + "
            f"{ready_index * replica_stride * _TILE_DEPENDENCY_COUNTER_STRIDE}"
        )
        publish = self._publish_tile_dependency_epoch(
            device_function,
            base=ready_base,
            epoch=epoch_var,
            replicas=epoch_replicas,
        )

        input_waits = [
            statement
            for counter, target in input_dependencies
            for statement in self._wait_for_counter(
                device_function=device_function,
                counter=counter,
                target=target,
                prefix="tile_dependency_singleton_input_wait",
            )
        ]
        singleton_body: list[ast.stmt] = [
            *input_waits,
            *_clone_opaque_statements(task_body),
            self._tile_dependency_publication_barrier(device_function),
            *publish,
        ]
        return [
            create(
                ast.If,
                test=expr_from_string(f"({worker}) == 0"),
                body=singleton_body,
                orelse=[],
            )
        ]

    def _emit_partitioned_dependency_pipeline(
        self,
        plan: _PartitionedDependencyPipeline,
        *,
        case_bodies: list[list[ast.stmt]],
        case_offsets: list[int],
        strategy: PersistentProgramIDs,
        device_function: DeviceFunction,
        worker: str,
        epoch_var: str,
        primary_arrivals: str,
        first_side_arrivals: str,
        second_side_arrivals: str,
        primary_ready: str,
        materialized_ready: str,
        partition_map_arrivals: str,
        reduction_arrivals: str,
        pipeline_ready: str,
        pipeline_epochs: str | None,
        plan_index: int,
        input_ready_counter: str | None = None,
        input_ready_target: str | None = None,
    ) -> list[ast.stmt]:
        logical_task = device_function.new_var(
            "tile_dependency_partitioned_producer_task", dce=True
        )
        group = device_function.new_var("tile_dependency_partition_group", dce=True)
        local = device_function.new_var("tile_dependency_partition_local", dce=True)
        member = device_function.new_var("tile_dependency_partition_member", dce=True)
        subtile = device_function.new_var("tile_dependency_partition_subtile", dce=True)
        physical_member = device_function.new_var(
            "tile_dependency_partition_physical_member", dce=True
        )
        physical_task = device_function.new_var(
            "tile_dependency_partition_physical_task", dce=True
        )
        primary_finalize_task = device_function.new_var(
            "tile_dependency_primary_finalize_task", dce=True
        )
        previous = device_function.new_var(
            "tile_dependency_partition_previous", dce=False
        )
        members_per_group = plan.primary_members_per_partition + 2
        tasks_per_group = members_per_group * plan.tiles_per_member
        primary_finalize_tasks = (
            plan.group_count
            * plan.primary_members_per_partition
            // plan.finalize_partition_block
        )
        primary_arrivals_per_finalizer = (
            plan.finalize_partition_block * plan.tiles_per_member
        )
        first_side_arrivals_per_cluster = (
            plan.finalize_partition_block * plan.tiles_per_member
        )
        primary_finalize_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.partition_finalize_root,
            logical_pid=(
                f"{case_offsets[plan.partition_finalize_root]} + "
                f"{primary_finalize_task}"
            ),
            body=case_bodies[plan.partition_finalize_root],
            name_suffix="primary",
            extra_argument_names=(primary_finalize_task,),
        )
        primary_finalizer = [
            primary_finalize_call,
            self._tile_dependency_publication_barrier(device_function),
            statement_from_string(
                f"tl.atomic_add({primary_ready} + {group}, {plan.finalize_partition_block}, sem='release', scope='gpu')"
            ),
        ]
        side_cluster = device_function.new_var("tile_dependency_side_cluster", dce=True)
        side_partition_local = device_function.new_var(
            "tile_dependency_side_partition_local", dce=True
        )
        side_partition = device_function.new_var(
            "tile_dependency_side_partition", dce=True
        )
        materialize_partition = device_function.new_var(
            "tile_dependency_materialize_partition", dce=True
        )
        materialize_tile_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.materialize_root,
            logical_pid=(
                f"{case_offsets[plan.materialize_root]} + {materialize_partition}"
            ),
            body=case_bodies[plan.materialize_root],
            extra_argument_names=(materialize_partition,),
        )
        side_partition_body: list[ast.stmt] = [
            statement_from_string(
                f"{side_partition} = {side_cluster} * {plan.finalize_partition_block} + {side_partition_local}"
            ),
            statement_from_string(f"{materialize_partition} = {side_partition}"),
        ]
        if plan.materialize_tasks_per_partition == 1:
            side_partition_body.extend(
                self._wait_for_counter(
                    device_function=device_function,
                    counter=f"{second_side_arrivals} + {materialize_partition}",
                    target=(
                        f"tl.cast({epoch_var}, tl.uint32) * "
                        f"tl.cast({plan.tiles_per_member}, tl.uint32)"
                    ),
                    prefix="tile_dependency_side_continuation_wait",
                )
            )
        side_partition_body.extend(
            (
                materialize_tile_call,
                self._tile_dependency_publication_barrier(device_function),
            )
        )
        if plan.materialize_tasks_per_partition == 1:
            side_partition_body.append(
                statement_from_string(
                    f"tl.atomic_xchg({materialized_ready} + {side_partition}, {epoch_var}, sem='release', scope='gpu')"
                )
            )
        else:
            side_partition_body.append(
                statement_from_string(
                    f"tl.atomic_add({materialized_ready} + {side_partition}, 1, sem='release', scope='gpu')"
                )
            )
        side_finalize_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.partition_finalize_root,
            logical_pid=(
                f"{case_offsets[plan.partition_finalize_root]} + "
                f"{primary_finalize_tasks} + {side_cluster}"
            ),
            body=case_bodies[plan.partition_finalize_root],
            name_suffix="side",
            extra_argument_names=(side_cluster,),
        )
        side_finalizer = [
            statement_from_string(
                f"{side_cluster} = {group} // {plan.finalize_partition_block}"
            ),
            side_finalize_call,
            self._tile_dependency_publication_barrier(device_function),
            create(
                ast.For,
                target=create(ast.Name, id=side_partition_local, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.static_range(0, {plan.finalize_partition_block})"
                ),
                body=side_partition_body,
                orelse=[],
                type_comment=None,
            ),
        ]
        consumer_affinity_order_mapping = [
            statement_from_string(f"{group} = {logical_task} // {tasks_per_group}"),
            statement_from_string(f"{local} = {logical_task} % {tasks_per_group}"),
            statement_from_string(f"{member} = {local} // {plan.tiles_per_member}"),
            statement_from_string(f"{subtile} = {local} % {plan.tiles_per_member}"),
            create(
                ast.If,
                test=expr_from_string(
                    f"{member} < {plan.primary_members_per_partition}"
                ),
                body=[
                    statement_from_string(
                        f"{physical_member} = {group} * {plan.primary_members_per_partition} + {member}"
                    )
                ],
                orelse=[
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{member} == {plan.primary_members_per_partition}"
                        ),
                        body=[
                            statement_from_string(
                                f"{physical_member} = {plan.group_count * plan.primary_members_per_partition} + {group}"
                            )
                        ],
                        orelse=[
                            statement_from_string(
                                f"{physical_member} = {plan.group_count * (plan.primary_members_per_partition + 1)} + {group}"
                            )
                        ],
                    )
                ],
            ),
        ]
        physical_mapping = [
            statement_from_string(
                f"{physical_member} = {logical_task} // {plan.tiles_per_member}"
            ),
            statement_from_string(
                f"{subtile} = {logical_task} % {plan.tiles_per_member}"
            ),
            create(
                ast.If,
                test=expr_from_string(
                    f"{physical_member} < {plan.group_count * plan.primary_members_per_partition}"
                ),
                body=[
                    statement_from_string(
                        f"{group} = {physical_member} // {plan.primary_members_per_partition}"
                    ),
                    statement_from_string(
                        f"{member} = {physical_member} % {plan.primary_members_per_partition}"
                    ),
                ],
                orelse=[
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{physical_member} < {plan.group_count * (plan.primary_members_per_partition + 1)}"
                        ),
                        body=[
                            statement_from_string(
                                f"{group} = {physical_member} - {plan.group_count * plan.primary_members_per_partition}"
                            ),
                            statement_from_string(
                                f"{member} = {plan.primary_members_per_partition}"
                            ),
                        ],
                        orelse=[
                            statement_from_string(
                                f"{group} = {physical_member} - {plan.group_count * (plan.primary_members_per_partition + 1)}"
                            ),
                            statement_from_string(
                                f"{member} = {plan.primary_members_per_partition + 1}"
                            ),
                        ],
                    )
                ],
            ),
        ]
        producer_tile_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.producer_root,
            logical_pid=f"{case_offsets[plan.producer_root]} + {physical_task}",
            body=case_bodies[plan.producer_root],
            extra_argument_names=(physical_task,),
        )
        if plan.materialize_tasks_per_partition == 1:
            second_side_body = [
                statement_from_string(
                    f"tl.atomic_add({second_side_arrivals} + {group}, 1, sem='acq_rel', scope='gpu')"
                )
            ]
        else:
            second_side_body = [
                statement_from_string(
                    f"{previous} = tl.atomic_add({second_side_arrivals} + {group}, 1, sem='acq_rel', scope='gpu')"
                ),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"({previous} % {plan.tiles_per_member}) == "
                        f"{plan.tiles_per_member - 1}"
                    ),
                    body=[
                        statement_from_string(
                            f"{materialize_partition} = {plan.group_count} + {group}"
                        ),
                        _clone_stmt(materialize_tile_call),
                        self._tile_dependency_publication_barrier(device_function),
                        statement_from_string(
                            f"tl.atomic_add({materialized_ready} + {group}, 1, sem='release', scope='gpu')"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        producer_body: list[ast.stmt] = [
            *(
                consumer_affinity_order_mapping
                if plan.consumer_affinity_order
                else physical_mapping
            ),
            statement_from_string(
                f"{physical_task} = {physical_member} * {plan.tiles_per_member} + {subtile}"
            ),
            producer_tile_call,
            self._tile_dependency_publication_barrier(device_function),
            create(
                ast.If,
                test=expr_from_string(
                    f"{member} < {plan.primary_members_per_partition}"
                ),
                body=[
                    statement_from_string(
                        f"{primary_finalize_task} = {physical_member} // {plan.finalize_partition_block}"
                    ),
                    statement_from_string(
                        f"{previous} = tl.atomic_add({primary_arrivals} + {primary_finalize_task}, 1, sem='acq_rel', scope='gpu')"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"({previous} % {primary_arrivals_per_finalizer}) == {primary_arrivals_per_finalizer - 1}"
                        ),
                        body=primary_finalizer,
                        orelse=[],
                    ),
                ],
                orelse=[
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{member} == {plan.primary_members_per_partition}"
                        ),
                        body=[
                            statement_from_string(
                                f"{side_cluster} = {group} // {plan.finalize_partition_block}"
                            ),
                            statement_from_string(
                                f"{previous} = tl.atomic_add({first_side_arrivals} + {side_cluster}, 1, sem='acq_rel', scope='gpu')"
                            ),
                            create(
                                ast.If,
                                test=expr_from_string(
                                    f"({previous} % {first_side_arrivals_per_cluster}) == {first_side_arrivals_per_cluster - 1}"
                                ),
                                body=side_finalizer,
                                orelse=[],
                            ),
                        ],
                        orelse=second_side_body,
                    )
                ],
            ),
        ]
        result: list[ast.stmt] = []
        if input_ready_counter is not None:
            assert input_ready_target is not None
            result.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{worker} < {plan.producer_tasks}"),
                    body=self._wait_for_counter(
                        device_function=device_function,
                        counter=input_ready_counter,
                        target=input_ready_target,
                        prefix="tile_dependency_producer_input_wait",
                    ),
                    orelse=[],
                )
            )
        result.append(
            create(
                ast.For,
                target=create(ast.Name, id=logical_task, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.range({worker}, {plan.producer_tasks}, {strategy.grid_size_expr})"
                ),
                body=producer_body,
                orelse=[],
                type_comment=None,
            )
        )

        partition_task = device_function.new_var(
            "tile_dependency_partition_task", dce=True
        )
        partition_index = device_function.new_var(
            "tile_dependency_partition_index", dce=True
        )
        reduction_task = device_function.new_var(
            "tile_dependency_reduction_task", dce=True
        )
        reduction_local = device_function.new_var(
            "tile_dependency_reduction_local", dce=True
        )
        reduction_subpartition = device_function.new_var(
            "tile_dependency_reduction_subpartition", dce=True
        )
        reduction_counter_index = device_function.new_var(
            "tile_dependency_reduction_counter_index", dce=True
        )
        final_reduction_task = device_function.new_var(
            "tile_dependency_final_reduction_task", dce=True
        )
        output_map_local = device_function.new_var(
            "tile_dependency_output_map_local", dce=True
        )
        output_maps_per_reduction = (
            plan.output_tasks_per_partition // plan.final_reduce_tasks_per_partition
        )
        pipeline_counter = (
            f"{pipeline_ready} + {plan_index * _TILE_DEPENDENCY_COUNTER_STRIDE}"
        )
        schedule = HostFunction.current().device_ir.tile_dependency_schedule
        assert schedule is not None and schedule.policy is not None
        epoch_replicas = schedule.policy.epoch_replicas or 1
        replicate_pipeline_completion = (
            pipeline_epochs is not None
            and plan.downstream_tasks > _TILE_DEPENDENCY_DIRECT_POLL_FANOUT_LIMIT
        )
        if replicate_pipeline_completion:
            pipeline_previous = device_function.new_var(
                "tile_dependency_pipeline_previous", dce=False
            )
            pipeline_epoch_base = (
                f"{pipeline_epochs} + "
                f"{plan_index * epoch_replicas * _TILE_DEPENDENCY_COUNTER_STRIDE}"
            )
            chain_completion = [
                statement_from_string(
                    f"{pipeline_previous} = tl.atomic_add({pipeline_counter}, 1, "
                    "sem='acq_rel', scope='gpu')"
                ),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"({pipeline_previous} % "
                        f"tl.cast({plan.completion_tasks}, tl.uint32)) == "
                        f"tl.cast({plan.completion_tasks - 1}, tl.uint32)"
                    ),
                    body=self._publish_tile_dependency_epoch(
                        device_function,
                        base=pipeline_epoch_base,
                        epoch=epoch_var,
                        replicas=epoch_replicas,
                    ),
                    orelse=[],
                ),
            ]
        else:
            chain_completion = [
                statement_from_string(
                    f"tl.atomic_add({pipeline_counter}, 1, sem='release', scope='gpu')"
                )
            ]
        output_map_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.map_root,
            logical_pid=(
                f"{case_offsets[plan.map_root]} + {partition_index} * "
                f"{plan.output_tasks_per_partition} + {reduction_local} * "
                f"{output_maps_per_reduction} + {output_map_local}"
            ),
            body=case_bodies[plan.map_root],
            extra_argument_names=(partition_index, reduction_local, output_map_local),
        )
        output_map_tail = [
            create(
                ast.For,
                target=create(ast.Name, id=output_map_local, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.static_range(0, {output_maps_per_reduction})"
                ),
                body=[output_map_call],
                orelse=[],
                type_comment=None,
            ),
            self._tile_dependency_publication_barrier(device_function),
            *chain_completion,
        ]
        reduction_tile_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.partition_reduce_root,
            logical_pid=(
                f"{case_offsets[plan.partition_reduce_root]} + {reduction_task}"
            ),
            body=case_bodies[plan.partition_reduce_root],
            extra_argument_names=(reduction_task,),
        )
        final_reduction_call: ast.stmt | None = None
        if plan.final_reduce_root is not None:
            final_reduction_call = self._outline_opaque_tile_body(
                device_function,
                root=plan.final_reduce_root,
                logical_pid=(
                    f"{case_offsets[plan.final_reduce_root]} + {final_reduction_task}"
                ),
                body=case_bodies[plan.final_reduce_root],
                extra_argument_names=(final_reduction_task,),
            )
        partition_continuation: list[ast.stmt] = [
            statement_from_string(f"{reduction_task} = {worker}"),
        ]
        if plan.final_reduce_root is None:
            partition_continuation.extend(
                [
                    statement_from_string(
                        f"{partition_index} = {reduction_task} // "
                        f"{plan.reduction_tasks_per_partition}"
                    ),
                    statement_from_string(
                        f"{reduction_local} = {reduction_task} % "
                        f"{plan.reduction_tasks_per_partition}"
                    ),
                    *self._wait_for_counter(
                        device_function=device_function,
                        counter=f"{partition_map_arrivals} + {partition_index}",
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({plan.partition_tasks_per_partition}, tl.uint32)"
                        ),
                        prefix="tile_dependency_partition_index_wait",
                    ),
                    reduction_tile_call,
                    self._tile_dependency_publication_barrier(device_function),
                    *[_clone_stmt(stmt) for stmt in output_map_tail],
                ]
            )
        else:
            assert final_reduction_call is not None
            arrivals_per_subpartition = (
                plan.partition_tasks_per_partition // plan.subpartitions_per_partition
            )
            reduction_previous = device_function.new_var(
                "tile_dependency_reduction_previous", dce=False
            )
            partition_continuation.extend(
                [
                    statement_from_string(
                        f"{final_reduction_task} = {reduction_task} // "
                        f"{plan.subpartitions_per_partition}"
                    ),
                    statement_from_string(
                        f"{partition_index} = {final_reduction_task} // "
                        f"{plan.final_reduce_tasks_per_partition}"
                    ),
                    statement_from_string(
                        f"{reduction_local} = {final_reduction_task} % "
                        f"{plan.final_reduce_tasks_per_partition}"
                    ),
                    statement_from_string(
                        f"{reduction_subpartition} = {reduction_task} % "
                        f"{plan.subpartitions_per_partition}"
                    ),
                    statement_from_string(
                        f"{reduction_counter_index} = {partition_index} * "
                        f"{plan.subpartitions_per_partition} + "
                        f"{reduction_subpartition}"
                    ),
                    *self._wait_for_counter(
                        device_function=device_function,
                        counter=(
                            f"{partition_map_arrivals} + {reduction_counter_index}"
                        ),
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({arrivals_per_subpartition}, tl.uint32)"
                        ),
                        prefix="tile_dependency_partition_subgroup_wait",
                    ),
                    reduction_tile_call,
                    self._tile_dependency_publication_barrier(device_function),
                    statement_from_string(
                        f"{reduction_previous} = "
                        f"tl.atomic_add({reduction_arrivals} + "
                        f"{final_reduction_task}, 1, sem='acq_rel', scope='gpu')"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"({reduction_previous} % "
                            f"{plan.subpartitions_per_partition}) == "
                            f"{plan.subpartitions_per_partition - 1}"
                        ),
                        body=[
                            final_reduction_call,
                            self._tile_dependency_publication_barrier(device_function),
                            *[_clone_stmt(stmt) for stmt in output_map_tail],
                        ],
                        orelse=[],
                    ),
                ]
            )
        partition_dependency = self._wait_for_counter(
            device_function=device_function,
            counter=f"{primary_ready} + {partition_index}",
            target=(
                f"tl.cast({epoch_var}, tl.uint32) * "
                f"tl.cast({plan.primary_members_per_partition}, tl.uint32)"
            ),
            prefix="tile_dependency_primary_ready_wait",
        )
        partition_dependency.extend(
            self._wait_for_counter(
                device_function=device_function,
                counter=f"{materialized_ready} + {partition_index}",
                target=(
                    f"tl.cast({epoch_var}, tl.uint32) * "
                    f"tl.cast({plan.materialize_tasks_per_partition}, tl.uint32)"
                ),
                prefix="tile_dependency_materialized_ready_wait",
            )
        )
        partition_tile_call = self._outline_opaque_tile_body(
            device_function,
            root=plan.partition_map_root,
            logical_pid=(f"{case_offsets[plan.partition_map_root]} + {partition_task}"),
            body=case_bodies[plan.partition_map_root],
            extra_argument_names=(partition_task,),
        )
        partition_body = [
            statement_from_string(f"{partition_task} = {worker}"),
            statement_from_string(
                f"{partition_index} = {partition_task} % {plan.group_count}"
            ),
            *partition_dependency,
            partition_tile_call,
            self._tile_dependency_publication_barrier(device_function),
        ]
        if plan.final_reduce_root is None:
            partition_body.append(
                statement_from_string(
                    f"tl.atomic_add({partition_map_arrivals} + "
                    f"{partition_index}, 1, sem='release', scope='gpu')"
                )
            )
        else:
            arrivals_per_subpartition = (
                plan.partition_tasks_per_partition // plan.subpartitions_per_partition
            )
            partition_body.extend(
                [
                    statement_from_string(
                        f"{reduction_subpartition} = "
                        f"({partition_task} // {plan.group_count}) // "
                        f"{arrivals_per_subpartition}"
                    ),
                    statement_from_string(
                        f"{reduction_counter_index} = {partition_index} * "
                        f"{plan.subpartitions_per_partition} + "
                        f"{reduction_subpartition}"
                    ),
                    statement_from_string(
                        f"tl.atomic_add({partition_map_arrivals} + "
                        f"{reduction_counter_index}, 1, sem='release', scope='gpu')"
                    ),
                ]
            )
        result.extend(
            [
                create(
                    ast.If,
                    test=expr_from_string(f"{worker} < {plan.partition_tasks}"),
                    body=partition_body,
                    orelse=[],
                ),
                create(
                    ast.If,
                    test=expr_from_string(f"{worker} < {plan.reduction_tasks}"),
                    body=partition_continuation,
                    orelse=[],
                ),
            ]
        )
        pipeline_dependency = (
            (
                (
                    f"{pipeline_epochs} + "
                    f"({plan_index * epoch_replicas} + "
                    f"(({worker}) % {epoch_replicas})) * "
                    f"{_TILE_DEPENDENCY_COUNTER_STRIDE}"
                ),
                f"tl.cast({epoch_var}, tl.uint32)",
            )
            if replicate_pipeline_completion
            else (
                pipeline_counter,
                (
                    f"tl.cast({epoch_var}, tl.uint32) * "
                    f"tl.cast({plan.completion_tasks}, tl.uint32)"
                ),
            )
        )
        result.append(
            create(
                ast.If,
                test=expr_from_string(f"{worker} < {plan.downstream_tasks}"),
                body=self._wait_for_counter(
                    device_function=device_function,
                    counter=pipeline_dependency[0],
                    target=pipeline_dependency[1],
                    prefix="tile_dependency_partition_chain_wait",
                ),
                orelse=[],
            )
        )
        return result


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


def _xcd_device_str() -> str:
    """Device expression for the current compile device (for host-side helpers)."""
    host_function = HostFunction.current()
    device = CompileEnvironment.current().device
    origins = [
        o for t, o in host_function.tensor_to_origin.items() if t.device == device
    ]
    if origins:
        return f"{origins[0].host_str()}.device"
    return f"torch.{device!r}"


def _maybe_emit_xcd_remap(
    device_function: DeviceFunction,
    pid_expr: str,
    total_expr: str,
    active_total_expr: str | None = None,
) -> tuple[list[ast.stmt], str]:
    """Optionally remap a program id into contiguous per-XCD regions.

    No-op (returns ``([], pid_expr)``) unless ``xcd_remap`` is enabled.  When
    enabled, emits the AITER-style contiguous-XCD remap (matching aiter
    ``remap_xcd``) and returns the name of the remapped variable.

    Used for:
    - ``flat`` / ``persistent_interleaved``: the (virtual) program id over the
      logical tile count;
    - ``persistent_blocked``: the worker id (``program_id(0)``) over the grid
      size, remapping which contiguous block each worker owns.
    """
    if not device_function.config.get("xcd_remap", False):
        return [], pid_expr

    # Inject _NUM_XCDS as a host-computed constexpr (mirrors _NUM_SM).
    if device_function.constexpr_arg(NUM_XCD_VAR):
        device_function.codegen.host_statements.append(
            statement_from_string(
                f"{NUM_XCD_VAR} = helion.runtime.get_num_xcd({_xcd_device_str()})"
            )
        )

    new_var = device_function.new_var
    pids_per = new_var("xcd_pids_per", dce=True)
    tall = new_var("xcd_tall", dce=True)
    xcd = new_var("xcd_id", dce=True)
    local = new_var("xcd_local_pid", dce=True)
    out = new_var("xcd_pid", dce=True)
    nx = NUM_XCD_VAR
    total = f"({active_total_expr or total_expr})"
    # Matches aiter remap_xcd: the first `tall` XCDs own one extra contiguous PID.
    stmts = [
        statement_from_string(f"{pids_per} = ({total} + {nx} - 1) // {nx}"),
        statement_from_string(f"{tall} = {total} % {nx}"),
        statement_from_string(f"{tall} = tl.where({tall} == 0, {nx}, {tall})"),
        statement_from_string(f"{xcd} = ({pid_expr}) % {nx}"),
        statement_from_string(f"{local} = ({pid_expr}) // {nx}"),
        statement_from_string(
            f"{out} = tl.where("
            f"{xcd} < {tall}, "
            f"{xcd} * {pids_per} + {local}, "
            f"{tall} * {pids_per} + ({xcd} - {tall}) * ({pids_per} - 1) + {local})"
        ),
    ]
    if active_total_expr is not None:
        guarded = new_var("xcd_pid", dce=True)
        stmts.append(
            statement_from_string(
                f"{guarded} = tl.where(({pid_expr}) < {total}, {out}, {total})"
            )
        )
        out = guarded
    return stmts, out


class FlatProgramIDs(ProgramIDs):
    """Only use the x grid and compute other dimensions"""

    def codegen(self, state: CodegenState) -> None:
        pid_var = self.shared_pid_var or typed_program_id(0)
        remap_stmts, pid_var = _maybe_emit_xcd_remap(
            state.device_function,
            pid_var,
            self.total_pids_expr(is_device=True),
        )
        statements = self._decompose_pid_to_statements(pid_var, state)
        state.codegen.statements_stack[-1][:] = [
            *remap_stmts,
            *statements,
            *state.codegen.statements_stack[-1],
        ]

    def codegen_grid(self) -> ast.AST:
        return expr_from_string(f"({self.total_pids_expr(is_device=False)},)")


@dataclasses.dataclass
class WorklistProgramIDs(ProgramIDs):
    """Compact-worklist grid: one program per work item.

    ``codegen`` emits ``_wid = pl.program_id(0)`` and recovers the owner
    coordinate from the ``owner_ids`` scalar-prefetch ref (``work_<owner>_ref[
    _wid]``) so owner-indexed tensors slice the right owner; ``codegen_grid``
    renders the **static** ``UPPER`` (megablocks bound) as the host grid
    positional, while the compact launcher overrides it with the traced
    ``num_work``.
    """

    upper_expr: str = "1"

    def codegen(self, state: CodegenState) -> None:
        from .pallas.compact_worklist import owner_ref_name

        env = CompileEnvironment.current()
        plan = env.compact_worklist_plan
        assert plan is not None
        stmts: list[ast.stmt] = [statement_from_string(f"_wid = {typed_program_id(0)}")]
        if self.pid_info:
            # owner_ids is always in the metadata (see metadata_arg_names): the
            # owner-grid prologue (q_offsets[seq]) is not DCE'd, so the owner pid
            # must be a valid owner index, NOT the work id (which ranges over
            # work items and would index q_offsets out of bounds).
            owner_pid = self.pid_info[0].pid_var
            ref = owner_ref_name(plan) + "_ref"
            stmts.append(statement_from_string(f"{owner_pid} = {ref}[_wid]"))
        state.codegen.statements_stack[-1][:] = [
            *stmts,
            *state.codegen.statements_stack[-1],
        ]

    def codegen_grid(self) -> ast.AST:
        return expr_from_string(f"({self.upper_expr},)")


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
        parent = self.parent_strategy
        parent_is_blocked = parent._is_persistent() and getattr(
            parent, "is_blocked", False
        )

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

        # xcd_remap (if enabled) regroups the PID space into contiguous per-XCD
        # regions *before* L2 grouping orders tiles within it.  For a blocked
        # persistent parent the remap is applied at the worker->block level in
        # the persistent setup instead, so skip it here to avoid double-remap.
        if parent_is_blocked:
            xcd_stmts: list[ast.stmt] = []
        else:
            xcd_stmts, pid = _maybe_emit_xcd_remap(
                state.device_function,
                pid,
                parent.total_pids_expr(is_device=True),
            )

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
            *xcd_stmts,
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
            stmts.append(
                statement_from_string(
                    f"{self.block_size_var} = {backend.cdiv_expr(self.total_pids_var, self.grid_size_expr, is_device=True)}"
                )
            )
            worker = typed_program_id(0)
            if DeviceFunction.current().config.get("xcd_remap", False):
                new_var = DeviceFunction.current().new_var
                safe_block_size = new_var("xcd_safe_block_size", dce=True)
                active_workers = new_var("xcd_active_workers", dce=True)
                stmts.extend(
                    [
                        statement_from_string(
                            f"{safe_block_size} = tl.maximum({self.block_size_var}, 1)"
                        ),
                        statement_from_string(
                            f"{active_workers} = {backend.cdiv_expr(self.total_pids_var, safe_block_size, is_device=True)}"
                        ),
                    ]
                )
                # Remap only the workers that own at least one block.  Slack
                # workers are sent past the end of the compact worker domain so
                # their persistent range is empty after end clipping.
                wstmts, worker = _maybe_emit_xcd_remap(
                    DeviceFunction.current(),
                    typed_program_id(0),
                    active_workers,
                    active_workers,
                )
                stmts.extend(wstmts)
            stmts.extend(
                [
                    statement_from_string(
                        f"{self.start_pid_var} = {worker} * {self.block_size_var}"
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
        # Interleaved: remap each virtual pid into per-XCD contiguous regions
        # (matches aiter's per-tile remap_xcd).  Blocked remaps the worker->block
        # assignment in the persistent setup instead, so it is skipped here.
        if not self.is_blocked:
            remap_stmts, pid_var = _maybe_emit_xcd_remap(
                state.device_function,
                pid_var,
                self.total_pids_expr(is_device=True),
            )
            setup_statements.extend(remap_stmts)
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

    _VALIDATED_TWO_CTA_MAX_K_TILES: ClassVar[int] = TCGEN05_TWO_CTA_MAX_K_TILES

    def __init__(self, *, is_blocked: bool) -> None:
        super().__init__(is_blocked=is_blocked)

    def _tcgen05_plan(self) -> CuteTcgen05MatmulPlan | None:
        try:
            return DeviceFunction.current().cute_state.matmul_plan
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

    def _tcgen05_cluster_n(self) -> int:
        # The tcgen05 plan owns ``cluster_n`` once the matmul plan has been
        # registered (cute_mma derives the validated value and stores it on
        # the plan). Outside the matmul codegen path the helper falls back
        # to the config knob; non-tcgen05 paths see cluster_n=1 from the
        # config default and never reach this method's result anyway.
        if (plan := self._tcgen05_plan()) is not None:
            return plan.cluster_n
        config = DeviceFunction.current().config
        cluster_n = int(str(config.get("tcgen05_cluster_n", 1)))
        return max(1, min(cluster_n, 2))

    def _tcgen05_l2_swizzle_size(self) -> int:
        """Return the L2 tile-scheduler swizzle size (Quack ``max_swizzle_size``).

        Returned value is the integer that will be threaded into
        ``cutlass.utils.PersistentTileSchedulerParams(swizzle_size=...)``.
        Default ``TCGEN05_L2_SWIZZLE_SIZE_DEFAULT`` (= ``1``) means no
        swizzle (preserves byte-identity vs the cycle 41 baseline).
        Larger values group consecutive cluster linear-IDs along the
        slow raster axis to promote L2 reuse on bandwidth-bound shapes.

        Mirrors ``_tcgen05_cluster_m`` / ``_tcgen05_cluster_n``: fall
        back to the legacy default when no matmul plan and no
        ``DeviceFunction`` are registered. Unit tests exercise the
        scheduler-prelude builders without a registered plan and
        expect the no-swizzle byte-identity path. Reads via the
        canonical ``l2_swizzle_size_from_config`` helper so
        codegen and the strategies layer share one decode.
        """
        if (plan := self._tcgen05_plan()) is not None:
            return plan.l2_swizzle_size
        try:
            config = DeviceFunction.current().config
        except NoCurrentFunction:
            return TCGEN05_L2_SWIZZLE_SIZE_DEFAULT
        return l2_swizzle_size_from_config(config)

    def _tcgen05_persistent_tile_sched_params_args(
        self, *, cluster_m: int, cluster_n: int
    ) -> str:
        """Format the constructor args for ``PersistentTileSchedulerParams``.

        Always passes the problem shape and cluster shape. When
        ``l2_swizzle_size > 1`` also passes the ``swizzle_size=`` kwarg
        so the CuTe scheduler folds in the L2 grouping math; when the
        size is ``1`` the kwarg is omitted to keep the no-swizzle path
        byte-identical to pre-cycle-42.

        Caller passes ``cluster_m`` / ``cluster_n`` so unit tests that
        exercise scheduler-prelude builders without a registered
        ``CuteTcgen05MatmulPlan`` (and without an active
        ``DeviceFunction``) can drive the helper from the
        ``_Tcgen05PersistentLayout`` they constructed locally.
        """
        problem = self._tcgen05_num_tiles_expr(is_device=True)
        l2_swizzle = self._tcgen05_l2_swizzle_size()
        if l2_swizzle <= 1:
            return f"{problem}, ({cluster_m}, {cluster_n}, 1)"
        return f"{problem}, ({cluster_m}, {cluster_n}, 1), swizzle_size={l2_swizzle}"

    def _tcgen05_is_two_cta(self) -> bool:
        if (plan := self._tcgen05_plan()) is not None:
            return plan.is_two_cta
        return False

    def _tcgen05_has_scheduler_warp(self) -> bool:
        plan = self._tcgen05_plan()
        return plan is not None and plan.has_scheduler_warp

    def _tcgen05_uses_grouped_static_persistent(self) -> bool:
        plan = self._tcgen05_plan()
        return bool(plan is not None and plan.grouped is not None)

    def _tcgen05_uses_grouped_worklist_nm_scheduler_mailbox(self) -> bool:
        plan = self._tcgen05_plan()
        return bool(
            plan is not None
            and plan.accumulator_view == "nm"
            and plan.has_scheduler_warp
            and plan.uses_role_local_persistent_body
        )

    def _tcgen05_sched_pipeline_plan(self) -> _Tcgen05SchedPipelinePlan | None:
        try:
            return DeviceFunction.current().cute_state.sched_pipeline_plan
        except NoCurrentFunction:
            return None

    def _tcgen05_sched_stage_count(self) -> int:
        plan = self._tcgen05_plan()
        if plan is None:
            return 0
        return max(plan.sched_stage_count, 0)

    def _tcgen05_uses_staged_work_tile_mailbox(self) -> bool:
        return self._tcgen05_sched_stage_count() > 1

    def _tcgen05_work_tile_slot_for_state(
        self,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
        i: int,
        pipeline_state: str | None,
    ) -> str:
        if pipeline_state is None:
            return f"{layout.work_tile_smem}[cutlass.Int32({i})]"
        return f"{layout.work_tile_smem}[cutlass.Int32({i}), {pipeline_state}.index]"

    def _tcgen05_work_tile_slot(
        self, layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout, i: int
    ) -> str:
        if not self._tcgen05_uses_staged_work_tile_mailbox():
            return self._tcgen05_work_tile_slot_for_state(layout, i, None)
        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        return self._tcgen05_work_tile_slot_for_state(
            layout, i, sched_plan.consumer_state
        )

    def _tcgen05_work_tile_producer_slot(
        self, layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout, i: int
    ) -> str:
        if not self._tcgen05_uses_staged_work_tile_mailbox():
            return self._tcgen05_work_tile_slot_for_state(layout, i, None)
        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        return self._tcgen05_work_tile_slot_for_state(
            layout, i, sched_plan.producer_state
        )

    def _tcgen05_work_tile_producer_smem_ptr(
        self, layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout
    ) -> str:
        if not self._tcgen05_uses_staged_work_tile_mailbox():
            return layout.work_tile_smem_ptr
        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        return (
            f"{layout.work_tile_smem}[None, {sched_plan.producer_state}.index].iterator"
        )

    def _tcgen05_work_tile_mailbox_field_count(self) -> int:
        if self._tcgen05_uses_grouped_worklist_nm_scheduler_mailbox():
            return TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT
        return 4

    def _tcgen05_has_validated_role_local_two_cta_runtime(self) -> bool:
        plan = self._tcgen05_plan()
        return bool(
            plan is not None
            and plan.is_two_cta
            and plan.uses_role_local_persistent_body
            and plan.k_tile_count <= self._VALIDATED_TWO_CTA_MAX_K_TILES
        )

    def _tcgen05_uses_cluster_m2_one_cta_role_local_bridge(self) -> bool:
        plan = self._tcgen05_plan()
        return bool(
            plan is not None
            and plan.uses_cluster_m2_one_cta_role_local_bridge
            and plan.cluster_m == 2
            and not plan.is_two_cta
            and plan.uses_role_local_persistent_body
        )

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
        # cluster_n>1 leaves the scheduler N dim equal to the logical N
        # tile count; the cluster_shape ``(cluster_m, cluster_n, 1)``
        # passed to ``PersistentTileSchedulerParams`` allocates one
        # cluster per ``cluster_n`` consecutive N tiles. Each CTA in the
        # cluster's N axis sees a distinct ``tile_idx[1]`` so the
        # virtual_pid mapping uses the raw scheduler tile_idx[1] as the
        # logical N coordinate.
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
        cluster_n = self._tcgen05_cluster_n()
        if cluster_m > 1:
            dims[0] = f"(({dims[0]}) + {cluster_m} - 1) // {cluster_m}"
        if cluster_n > 1:
            # Each cluster covers ``cluster_n`` consecutive logical N tiles.
            dims[1] = f"(({dims[1]}) + {cluster_n} - 1) // {cluster_n}"
        return " * ".join(f"({dim})" for dim in dims[:3])

    def _tcgen05_max_persistent_work_clusters_expr(self) -> str:
        """Return the launch-grid persistent work-cluster capacity.

        Capacity is in cluster slots; divide by ``cluster_m`` (V-pair
        size) so each cluster slot consumes one SM regardless of
        ``cluster_n``. Independent of ``cluster_n`` so cluster_n=2
        does not collapse the launch to one wave.
        """
        cluster_m = self._tcgen05_cluster_m()
        cluster_n = self._tcgen05_cluster_n()
        if cluster_m * cluster_n == 1:
            return self.grid_size_expr
        return f"max(1, ({self.grid_size_expr}) // {cluster_m})"

    def _tcgen05_grid_work_clusters_expr(self, total_clusters: str) -> str:
        """Return the scheduler z dimension for the persistent launch grid."""
        max_persistent_clusters = self._tcgen05_max_persistent_work_clusters_expr()
        return f"min(({total_clusters}), ({max_persistent_clusters}))"

    def codegen_grid(self) -> ast.AST:
        # Tcgen05 persistent kernels use CUTLASS' z-indexed scheduler instead
        # of the parent virtual-PID loop. Validated role-local CtaGroup.TWO
        # caps the launch at persistent work-cluster capacity. Validated
        # role-local CtaGroup.TWO uses per-role scheduler loops over this same
        # capped grid, so it can recycle CTA-local pipeline/TMEM state across
        # logical work tiles. Guarded legacy fallback and K-over-cap
        # CtaGroup.TWO use the same capped grid but still raise before launch.
        # Multi-root ForEach kernels are still host-guarded because this grid
        # is derived from this case's pid_info only.
        cluster_m = self._tcgen05_cluster_m()
        cluster_n = self._tcgen05_cluster_n()
        total_clusters = self._tcgen05_num_work_clusters_expr(is_device=False)
        plan = self._tcgen05_plan()
        if plan is not None and plan.is_clc_persistent:
            # G2-H (cute_plan.md): CLC mode launches the *full*
            # problem grid (one cluster slot per problem cluster),
            # not the persistent sub-grid. The hardware tile-scheduler
            # then controls which clusters actually run; CLC's
            # ``try_cancel`` lets a running cluster cancel and steal
            # work from a not-yet-started cluster. Mirrors Quack's
            # ``get_grid_shape`` for ``PersistenceMode.CLC`` and
            # cutlass-DSL's ``ClcDynamicPersistentTileScheduler.get_grid_shape``
            # which both return the full problem grid for CLC mode.
            # Capping the launch like the static path does (``min(total,
            # max_persistent)``) starves the hardware of pending
            # clusters and causes CLC to immediately return ``valid=0``
            # on the first query, terminating the persistent loop
            # after iteration 0 (verified via ``cute.printf``).
            return expr_from_string(f"({cluster_m}, {cluster_n}, {total_clusters})")
        grid_work_clusters = self._tcgen05_grid_work_clusters_expr(total_clusters)
        return expr_from_string(f"({cluster_m}, {cluster_n}, {grid_work_clusters})")

    def _tcgen05_logical_m_coord_expr(self, coord: str) -> str:
        if self._tcgen05_is_two_cta():
            return f"({coord}) // cutlass.Int32({self._tcgen05_cluster_m()})"
        if self._tcgen05_uses_cluster_m2_one_cta_role_local_bridge():
            # The shared clustered scheduler publishes ``base_m + peer_rank``
            # into each CTA's SMEM slot. The guarded role-local bridge omits
            # that handoff, so bind each CTA's role-local PID to the same
            # per-peer M coordinate directly.
            return (
                f"({coord}) + "
                "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())"
            )
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

    def _tcgen05_output_full_tile_expr_for_work_tile(self, work_tile_var: str) -> str:
        """Return whether a scheduler work tile covers a full output tile.

        Used by the scheduler-backed edge path to publish interior tiles and
        fringe tiles through separate scheduler phases while keeping every
        consumer role on the same tile order within each phase. The predicate
        must match the consumer's post-L2-remap ``pid_0`` / ``pid_1`` rather
        than the scheduler's raw tile coordinates; otherwise grouped PID order
        can send a fringe tile down the full-tile TMA-store path.
        """
        assert len(self.pid_info) >= 2, (
            "tcgen05 output full-tile split requires M/N PID dimensions"
        )

        def pid_numel_expr(pid: PIDInfo) -> str:
            if isinstance(pid.numel, str):
                return pid.numel
            return DeviceFunction.current().sympy_expr(pid.numel)

        def l2_grouping() -> int:
            raw = DeviceFunction.current().config.get("l2_groupings", [1])
            if isinstance(raw, (list, tuple)):
                return int(str(raw[0])) if raw else 1
            return int(str(raw))

        # M and N are the trailing two PID axes; any leading axes are
        # passthrough (batch) that only offset memory (mirrors
        # ``_specialized_mma_root_mn_block_ids``), so read M/N from the tail
        # rather than positions 0/1. NOTE: the coord extraction below linearizes
        # over M/N only. Batched *partial*-tile edge splitting would also need
        # the leading passthrough factored out of the virtual pid; that is not a
        # validated path (the multi-tile guard restricts batched 2-CTA to static
        # full tiles, for which this predicate is true for every tile anyway).
        m_pid = self.pid_info[-2]
        n_pid = self.pid_info[-1]
        virtual_pid = self._tcgen05_linear_virtual_pid_expr(work_tile_var)
        num_pid_m = m_pid.num_pids_expr(is_device=True)
        l2_group = l2_grouping()
        if l2_group > 1:
            num_pid_n = n_pid.num_pids_expr(is_device=True)
            num_pid_in_group = f"cutlass.Int32({l2_group}) * ({num_pid_n})"
            group_id = f"({virtual_pid}) // ({num_pid_in_group})"
            first_pid_m = f"({group_id}) * cutlass.Int32({l2_group})"
            group_size_m = (
                f"min(({num_pid_m}) - ({first_pid_m}), cutlass.Int32({l2_group}))"
            )
            m_coord = (
                f"({first_pid_m}) + "
                f"((({virtual_pid}) % ({num_pid_in_group})) % ({group_size_m}))"
            )
            n_coord = f"(({virtual_pid}) % ({num_pid_in_group})) // ({group_size_m})"
        else:
            m_coord = f"({virtual_pid}) % ({num_pid_m})"
            n_coord = f"({virtual_pid}) // ({num_pid_m})"

        m_extent = pid_numel_expr(m_pid)
        n_extent = pid_numel_expr(n_pid)
        return (
            f"({m_coord}) * ({m_pid.block_size_var}) "
            f"+ ({m_pid.block_size_var}) <= ({m_extent}) "
            "and "
            f"({n_coord}) * ({n_pid.block_size_var}) "
            f"+ ({n_pid.block_size_var}) <= ({n_extent})"
        )

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
        a single, always-launched warp, so it is a stable owner for the shared
        scheduler prelude and for any residual shared loop kept by validated
        cluster_m=1 or guarded fallback shapes.
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
        """Boolean expression that gates the epilogue warps' role block.

        Workstream A Stage 4 (cycle 93, Path B shared-loop split): when the
        matmul plan carries a store warp (``has_store_warp``), the predicate
        is WIDENED to also admit ``store_warp_id`` so the store warp runs the
        SAME epilogue role-local while as the 4 epi warps — sharing the
        descriptor/SMEM/layout setup and the sched-consumer + subtile loop
        (no re-derivation; this is what makes Path B cheap vs the independent
        store-warp loop of Path A). Inside the loop the per-subtile tail is
        split by warp role with inline gates emitted in the tail source
        (``memory_ops._codegen_cute_store_tcgen05_tile``): the 4 epi warps
        (``epi_active`` = ``warp_idx < epi_warp_count``) own T2R/R2S + the
        C-store producer commit, and the store warp (``warp_idx ==
        store_warp_id``) owns the consumer-wait + TMA-D + consumer-release
        drain. When ``store_warps=0`` the predicate is the historical
        ``warp_idx < epi_warp_count`` and codegen is byte-identical.
        """
        plan = self._tcgen05_plan()
        assert plan is not None, (
            "tcgen05 epilogue role predicate requires a registered matmul plan"
        )
        epi_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"< cutlass.Int32({plan.epi_warp_count})"
        )
        if not plan.has_store_warp:
            return epi_predicate
        return (
            f"({epi_predicate} or "
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.store_warp_id}))"
        )

    def _tcgen05_scheduler_role_predicate(self) -> str:
        """Boolean expression that gates the scheduler warp's role block.

        Active under ``ROLE_LOCAL_WITH_SCHEDULER`` only; gating is
        ``warp_idx == scheduler_warp_id`` against the dedicated warp
        the matmul plan reserves for centralized tile scheduling.
        """
        plan = self._tcgen05_plan()
        assert plan is not None and plan.has_scheduler_warp, (
            "tcgen05 scheduler role predicate requires a matmul plan "
            "with has_scheduler_warp=True"
        )
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            "== cutlass.Int32(plan.scheduler_warp_id)".replace(
                "plan.scheduler_warp_id", str(plan.scheduler_warp_id)
            )
        )

    def _tcgen05_c_input_role_predicate(self) -> str:
        """Boolean expression that gates the C-input warp's role block.

        Active when the matmul plan has ``c_input_warp_count > 0``
        (``cute_plan.md`` §7.5.3.2). The C-input warp sits at warp
        id ``scheduler_warp_id + scheduler_warp_count`` — directly
        after the scheduler warp in the launched-CTA layout. Cycle 1
        of the producer-body split: the role-local while body is
        empty (consumer side still reads from GMEM in
        ``memory_ops._aux_subtile_load_source``); cycle 2 fills in
        the producer GMEM→SMEM cooperative copy, cycle 3 the
        consumer-side SMEM read flip.
        """
        plan = self._tcgen05_plan()
        assert plan is not None and plan.has_c_input_warp, (
            "tcgen05 C-input role predicate requires a matmul plan "
            "with has_c_input_warp=True"
        )
        return (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
            f"== cutlass.Int32({plan.c_input_warp_id})"
        )

    def _split_tcgen05_invariant_setup(
        self, device_function: DeviceFunction, body: list[ast.stmt]
    ) -> tuple[list[ast.stmt], list[ast.stmt], list[ast.stmt]]:
        """Split the device-function prefix into hoisted setup vs per-tile body.

        Codegen has explicitly tagged the per-tile statements via
        ``cute_state.register_tcgen05_per_tile_stmts``. Everything else can be
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
        cute_state = device_function.cute_state
        if not cute_state.has_tcgen05_per_tile_marks:
            return [], [], body

        per_tile_names: set[str] = {self.virtual_pid_var}
        hoisted: list[ast.stmt] = []
        epi_role_prelude: list[ast.stmt] = []
        wrapped: list[ast.stmt] = []
        for stmt in body:
            if cute_state.is_tcgen05_epi_role_prelude(stmt):
                epi_role_prelude.append(stmt)
                continue
            reads, writes = _stmt_name_uses(stmt)
            is_per_tile = (
                cute_state.is_tcgen05_per_tile(stmt)
                or bool(reads & per_tile_names)
                or bool(writes & per_tile_names)
            )
            if is_per_tile:
                per_tile_names.update(writes)
                wrapped.append(stmt)
            else:
                hoisted.append(stmt)
        return hoisted, epi_role_prelude, wrapped

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
        cute_state = device_function.cute_state
        role_predicates_by_id: dict[int, str] = {}
        for stmt_id in cute_state.tcgen05_tma_load_role_stmt_ids:
            role_predicates_by_id[stmt_id] = tma_load_predicate
        for stmt_id in cute_state.tcgen05_mma_exec_role_stmt_ids:
            assert stmt_id not in role_predicates_by_id, (
                "tcgen05 role statement registered for multiple warp roles"
            )
            role_predicates_by_id[stmt_id] = mma_exec_predicate
        for stmt_id in cute_state.tcgen05_epi_role_stmt_ids:
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

        Statements registered via ``cute_state.register_tcgen05_post_loop_stmts``
        belong after the persistent work-tile loop (one-shot drains:
        ``producer_tail``, TMEM dealloc, allocator setup). Without this
        extraction they would execute every tile, which wastes work and
        can corrupt pipeline state.
        """
        cute_state = device_function.cute_state
        if not cute_state.has_tcgen05_post_loop_marks:
            return body, []
        remaining: list[ast.stmt] = []
        post_loop: list[ast.stmt] = []
        for stmt in body:
            if cute_state.is_tcgen05_post_loop(stmt):
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
        "Helion CuTe persistent + tcgen05 currently supports runtime "
        "execution only for validated single-root static full tiles: "
        "tcgen05_cluster_m=1 or role-local CtaGroup.TWO "
        "tcgen05_cluster_m=2 with at most 256 K tiles. Partial K/M/N tile "
        "fallback shapes, CtaGroup.TWO shapes above the validated K-tile "
        "limit, multi-root kernels, and unvalidated cluster_m settings can "
        "produce wrong output, hang, or launch-fail. The kernel was launched "
        "with total_tiles=%d, which is outside the validated persistent "
        "scheduler set for this path. "
        'Use a non-persistent pid_type (e.g. "flat"), pick a single-root '
        "static-full-tile kernel with tcgen05_cluster_m=1, or pick a "
        "validated single-root static-full CtaGroup.TWO shape with at most "
        "256 K tiles."
    )

    def _emit_host_multi_tile_guard(
        self,
        device_function: DeviceFunction,
        host_total_pids_expr: str | None = None,
        guard_threshold: int | str = 1,
    ) -> None:
        """Emit a host-side guard against multi-tile execution.

        The single-root static full-tile role-local path has multi-tile
        runtime coverage for ``tcgen05_cluster_m == 1``. Validated static-full
        CtaGroup.TWO uses role-local scheduler loops over the capped persistent
        grid, so no multi-tile host guard is emitted for that set.
        Legacy non-role-local tcgen05 persistent kernels, multi-root kernels,
        cluster_m > 1 fallback configs, and CtaGroup.TWO configs above the
        validated K-tile cap still hit, or lack coverage for, wrong-output /
        hang / launch-failure modes, so this guard remains for those paths.
        Single-tile cluster_m=1 fallback shapes continue to run.

        The autotuner narrowing in
        ``ConfigSpec.narrow_tcgen05_autotune_to_validated_configs`` removes
        ``persistent_blocked`` / ``persistent_interleaved`` from the search
        space for tcgen05 BF16/FP16 matmuls, so this guard only fires for
        explicit user configs that bypass autotune.

        The threshold is intentionally ``total_tiles > 1`` for guarded
        cluster_m=1 single-root fallback paths. For cluster_m > 1 fallback and
        CtaGroup.TWO shapes above the K-tile cap this converts known launch,
        timeout, and wrong-output failures into a host error. Multi-root kernels use
        ``total_tiles > 0`` because the scheduler grid is derived from only the
        first root case; even one tile in a later case is unsafe.
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
        hoisted_setup, epi_role_prelude_stmts, wrapped_body = (
            self._split_tcgen05_invariant_setup(device_function, wrapped_body)
        )

        layout = self._build_tcgen05_persistent_layout(device_function)
        partition = self._partition_tcgen05_role_blocks(device_function, wrapped_body)
        use_role_local_body = bool(partition.role_blocks_extracted)
        role_local_predicates = {
            role_block.role_predicate
            for role_block in partition.role_blocks_extracted
            if role_block.role_predicate is not None
        }
        has_all_role_local_bodies = {
            self._tcgen05_tma_load_role_predicate(),
            self._tcgen05_mma_exec_role_predicate(),
            self._tcgen05_epi_role_predicate(),
        }.issubset(role_local_predicates)
        use_validated_cluster_m1_role_local_body = (
            use_role_local_body and layout.cluster_m == 1 and not is_multi_root
        )
        use_validated_two_cta_role_local_body = (
            has_all_role_local_bodies
            and layout.cluster_m == 2
            and self._tcgen05_has_validated_role_local_two_cta_runtime()
            and not is_multi_root
        )
        use_validated_role_local_body = (
            use_validated_cluster_m1_role_local_body
            or use_validated_two_cta_role_local_body
        )
        # Once all three work-producing roles own independent persistent
        # schedulers, a single-root kernel no longer needs the shared scheduler
        # for progress. Whether its loop can actually disappear is decided
        # separately from the generated residual body's meaningful work.
        can_omit_shared_scheduler = has_all_role_local_bodies and not is_multi_root
        omit_shared_loop = (
            can_omit_shared_scheduler
            and not self._tcgen05_shared_loop_has_meaningful_work(
                partition, post_loop_stmts
            )
        )
        if self._tcgen05_uses_staged_work_tile_mailbox() and not omit_shared_loop:
            raise exc.InvalidConfig(
                f"{TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY}=2 requires omitted "
                "shared-loop full role-local scheduler codegen"
            )
        if use_role_local_body:
            # Retarget even for guarded cluster_m>1 / multi-root codegen so
            # compile-only inspection still sees the role-local scheduler shape.
            self._retarget_tcgen05_shared_scheduler_to_exec(layout)
        if not use_validated_role_local_body:
            guard_threshold: int | str
            if is_multi_root:
                guard_threshold = 0
            elif layout.cluster_m == 1:
                guard_threshold = 1
            else:
                # cluster_m > 2, cluster_m=2 without the full role-local body,
                # and role-local CtaGroup.TWO above the K-tile cap all use the
                # strict guard.
                guard_threshold = 0
            self._emit_host_multi_tile_guard(
                device_function,
                host_guard_total_pids,
                guard_threshold=guard_threshold,
            )

        setup: list[ast.stmt] = []
        # Fully role-local codegen does not consume the shared work-tile SMEM
        # handoff. Each role owns a scheduler loop over the capped persistent
        # grid, so validated cluster_m=1 and CtaGroup.TWO skip the shared
        # scheduler and residual loop.
        if not omit_shared_loop:
            setup.extend(self._build_tcgen05_persistent_prelude(layout))
        elif self._tcgen05_has_scheduler_warp():
            # ``ROLE_LOCAL_WITH_SCHEDULER`` skips the shared loop but
            # *does* need the per-CTA work-tile SMEM mailbox: the
            # scheduler warp publishes per-tile coords there and
            # consumer warps read them after ``consumer_wait``.
            setup.extend(
                self._build_tcgen05_work_tile_smem_alloc(layout, staged_ok=True)
            )
        setup.extend(hoisted_setup)
        if use_role_local_body:
            if omit_shared_loop:
                role_local_whiles, shared_tile_body = (
                    self._build_tcgen05_persistent_tile_body_role_local(
                        device_function,
                        layout,
                        partition,
                        build_shared_tile_body=False,
                        epi_role_prelude_stmts=epi_role_prelude_stmts,
                        post_loop_stmts=post_loop_stmts,
                    )
                )
            else:
                role_local_whiles, shared_tile_body = (
                    self._build_tcgen05_persistent_tile_body_role_local(
                        device_function,
                        layout,
                        partition,
                        epi_role_prelude_stmts=epi_role_prelude_stmts,
                    )
                )
            setup.extend(role_local_whiles)
            if not omit_shared_loop:
                # Partial and multi-root role-local shapes still rejoin the
                # shared loop. Validated fully role-local codegen skips this
                # residual loop; its work is already owned by role-local
                # schedulers and cross-role pipelines.
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
        # ``cluster_n`` is the multicast factor along the cluster N axis;
        # default 1 keeps every existing test (and the cluster_n=1 byte-
        # identity golden) unchanged. Threaded through to the launch grid
        # / scheduler params when cluster_n>1 (cute_plan.md §6.12.7).
        cluster_n: int = 1

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
                # The shared-loop scheduler bridge remains one-stage: its
                # mailbox is a single 4-Int32 tuple, so the producer arms the
                # consumer-state full mbarrier before the remote stores.
                # Staged mailboxes are only used after the shared loop is
                # omitted in the CLC role-local scheduler path.
                statement_from_string(
                    f"{sched_barrier_ptr} = "
                    f"{sched_pipeline}.producer_get_barrier({sched_consumer_state})"
                ),
                statement_from_string(f"{sched_peer_rank} = cute.arch.lane_idx()"),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{sched_peer_rank} < cutlass.Int32({cluster_m})"
                    ),
                    body=[
                        statement_from_string(f"{sched_peer_m} = {sched_peer_rank}"),
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
            cluster_n=self._tcgen05_cluster_n(),
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

    def _build_tcgen05_work_tile_smem_alloc(
        self, layout: _Tcgen05PersistentLayout, *, staged_ok: bool = False
    ) -> list[ast.stmt]:
        """Allocate the per-CTA work-tile SMEM mailbox.

        This is the 4-Int32 work-tile tuple, optionally repeated per
        scheduler stage, used to broadcast tile coordinates + an
        is-valid sentinel. Both the cluster_m=2 ONE-CTA bridge path
        and ``ROLE_LOCAL_WITH_SCHEDULER`` use this storage, so the
        allocation is pulled out of
        ``_build_tcgen05_persistent_prelude`` (which is conditionally
        skipped when the residual shared loop is omitted) into its
        own helper that always runs when the work-tile mailbox is
        needed.
        """
        field_count = self._tcgen05_work_tile_mailbox_field_count()
        if self._tcgen05_uses_staged_work_tile_mailbox():
            assert staged_ok, (
                "staged work-tile mailbox requires omitted shared-loop "
                "role-local scheduler codegen"
            )
            plan = self._tcgen05_plan()
            assert plan is not None and plan.is_clc_persistent and plan.cluster_m > 1, (
                "staged work-tile mailbox is only validated for clustered CLC"
            )
            stage_count = self._tcgen05_sched_stage_count()
            alloc_extent = f"cutlass.Int32({field_count * stage_count})"
            layout_expr = (
                f"cute.make_layout(({field_count}, {stage_count}), "
                f"stride=(1, {field_count}))"
            )
        else:
            alloc_extent = str(field_count)
            layout_expr = f"cute.make_layout(({field_count},), stride=(1,))"
        return [
            statement_from_string(
                f"{layout.work_tile_smem_ptr} = cute.arch.alloc_smem("
                f"cutlass.Int32, {alloc_extent}, alignment=16)"
            ),
            statement_from_string(
                f"{layout.work_tile_smem_tensor} = cute.make_tensor("
                f"{layout.work_tile_smem_ptr}, {layout_expr})"
            ),
            statement_from_string(
                f"{layout.work_tile_smem} = {layout.work_tile_smem_tensor}"
            ),
        ]

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
                f"{self._tcgen05_persistent_tile_sched_params_args(cluster_m=layout.cluster_m, cluster_n=layout.cluster_n)})"
            ),
            statement_from_string(
                f"{layout.tile_sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                f"{layout.tile_sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
            ),
            *self._build_tcgen05_work_tile_smem_alloc(layout),
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
        role_prelude_stmts: list[ast.stmt] | None = None,
        *,
        emit_pdl_wait: bool = True,
        initialize_tile_counter: bool = True,
        store_aux_per_tile_stmts: list[ast.stmt] | None = None,
        store_aux_predicate: str | None = None,
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
        the role-local loop. The caller decides whether to append a residual
        shared loop after these role-local loops. It is omitted when the
        residual shared body contains only cloned dependency setup and legacy
        barriers that no longer protect shared work.

        The returned statement is the role-local ``while`` itself,
        wrapped in ``if {role_predicate}:`` so only the matching warps
        enter the loop. The caller appends this statement inside the
        persistent kernel's setup list.

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

        # ``ROLE_LOCAL_WITH_SCHEDULER`` reroutes the per-role body
        # through the broadcast pipeline. When active, the consumer
        # warp waits on the sched_pipeline, reads the published tile
        # metadata from SMEM, releases the sched stage, then runs its
        # role block.
        # The per-role ``StaticPersistentTileScheduler.create`` is
        # *not* emitted in this mode — the scheduler warp owns the
        # only tile scheduler.
        if self._tcgen05_uses_grouped_worklist_nm_scheduler_mailbox():
            return self._build_grouped_static_role_local_while_with_scheduler(
                device_function,
                layout,
                role_block,
                scheduler_var_prefix=scheduler_var_prefix,
                dependency_stmts=dependency_stmts,
                role_prelude_stmts=role_prelude_stmts,
                emit_pdl_wait=emit_pdl_wait,
                initialize_tile_counter=initialize_tile_counter,
            )
        if self._tcgen05_has_scheduler_warp():
            return self._build_role_local_while_with_scheduler(
                device_function,
                layout,
                role_block,
                scheduler_var_prefix=scheduler_var_prefix,
                dependency_stmts=dependency_stmts,
                role_prelude_stmts=role_prelude_stmts,
                emit_pdl_wait=emit_pdl_wait,
                initialize_tile_counter=initialize_tile_counter,
                store_aux_per_tile_stmts=store_aux_per_tile_stmts,
                store_aux_predicate=store_aux_predicate,
            )
        if self._tcgen05_uses_grouped_static_persistent():
            return self._build_grouped_static_role_local_while(
                device_function,
                role_block,
                scheduler_var_prefix=scheduler_var_prefix,
                dependency_stmts=dependency_stmts,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
        assert store_aux_per_tile_stmts is None, (
            "store-warp aux merge requires ROLE_LOCAL_WITH_SCHEDULER"
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

        prelude: list[ast.stmt] = []
        if (
            emit_pdl_wait
            and self._tcgen05_is_two_cta()
            and role_block.role_predicate == self._tcgen05_tma_load_role_predicate()
        ):
            # PDL parity with Quack/CUTLASS: TMA producers wait before
            # touching scheduler state or issuing global-memory TMA work.
            prelude.append(statement_from_string("cute.arch.griddepcontrol_wait()"))
        prelude.extend(
            [
                statement_from_string(
                    f"{sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                    f"{self._tcgen05_persistent_tile_sched_params_args(cluster_m=layout.cluster_m, cluster_n=layout.cluster_n)})"
                ),
                statement_from_string(
                    f"{sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                    f"{sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
                ),
                statement_from_string(
                    f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
                ),
            ]
        )
        tile_counter_var, increment_tile_counter_per_tile = (
            self._finish_role_local_prelude(
                device_function,
                role_block,
                prelude,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
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
        if tile_counter_var is not None and increment_tile_counter_per_tile:
            per_tile_body.append(
                statement_from_string(
                    f"{tile_counter_var} = {tile_counter_var} + cutlass.Int32(1)"
                )
            )
        if (plan := self._tcgen05_plan()) is not None and plan.one_shot_role_scheduler:
            prelude.extend(per_tile_body)
        else:
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

    def _finish_role_local_prelude(
        self,
        device_function: DeviceFunction,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        prelude: list[ast.stmt],
        *,
        role_prelude_stmts: list[ast.stmt] | None,
        initialize_tile_counter: bool,
    ) -> tuple[str | None, bool]:
        tile_counter_var = (
            device_function.cute_state.epi_role_tile_counter_var
            if role_block.role_predicate == self._tcgen05_epi_role_predicate()
            else None
        )
        increment_tile_counter_per_tile = (
            device_function.cute_state.epi_role_tile_counter_increment_per_tile
            if tile_counter_var is not None
            else False
        )
        if tile_counter_var is not None and initialize_tile_counter:
            prelude.append(
                statement_from_string(f"{tile_counter_var} = cutlass.Int32(0)")
            )
        prelude.extend(role_prelude_stmts or ())
        return tile_counter_var, increment_tile_counter_per_tile

    @staticmethod
    def _grouped_static_dependency_stmts(
        dependency_stmts: list[ast.stmt] | None,
    ) -> list[ast.stmt]:
        if dependency_stmts is None:
            return []
        grouped_coord_names = {
            "virtual_pid",
            "pid_0",
            "pid_1",
            "tile_offset_0",
            "tile_offset_1",
        }
        filtered: list[ast.stmt] = []
        for stmt in dependency_stmts:
            _reads, writes = _stmt_name_uses(stmt)
            if not (writes & grouped_coord_names):
                filtered.append(stmt)
                continue
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and writes <= grouped_coord_names
            ):
                continue
            raise AssertionError(
                "tcgen05 grouped static scheduler cannot drop mixed coordinate "
                "dependency statement: " + ast.unparse(stmt)
            )
        return filtered

    def _grouped_worklist_nm_valid_store_m_stmts(
        self,
        device_function: DeviceFunction,
    ) -> list[ast.stmt]:
        plan = self._tcgen05_plan()
        assert plan is not None and plan.accumulator_view == "nm"
        grouped = plan.grouped
        assert grouped is not None
        assert grouped.valid_m is not None and grouped.store_m is not None
        grouped_valid_m = grouped.valid_m
        grouped_store_m = grouped.store_m
        grouped_layout = grouped.layout
        grouped_metadata_idx = grouped.metadata_idx
        grouped_cta_tile_idx_m = grouped.cta_tile_idx_m

        def metadata_load_expr(column: int) -> str:
            return (
                f"({grouped_layout}.iterator + "
                f"{grouped_metadata_idx} * "
                f"cutlass.Int32({grouped_layout}.layout.stride[0]) + "
                f"cutlass.Int32({column}) * "
                f"cutlass.Int32({grouped_layout}.layout.stride[1])).load()"
            )

        tile_start_var = device_function.new_var("tcgen05_grouped_selected_tile_start")
        source_tile_m = plan.source_tile_m
        if grouped.device_split_sizes:
            remaining_m = (
                f"max({grouped.problem_m} - {tile_start_var}, cutlass.Int32(0))"
            )
            return [
                statement_from_string(
                    f"{tile_start_var} = {grouped_cta_tile_idx_m} * "
                    f"cutlass.Int32({source_tile_m})"
                ),
                statement_from_string(
                    f"{grouped_valid_m} = min(cutlass.Int32({source_tile_m}), "
                    f"{remaining_m})"
                ),
                statement_from_string(
                    f"{grouped_store_m} = min(cutlass.Int32({source_tile_m}), "
                    f"{remaining_m})"
                ),
            ]
        return [
            statement_from_string(
                f"{tile_start_var} = {grouped_cta_tile_idx_m} * "
                f"cutlass.Int32({source_tile_m})"
            ),
            statement_from_string(
                f"{grouped_valid_m} = min(cutlass.Int32({source_tile_m}), "
                f"max({metadata_load_expr(2)} - {tile_start_var}, cutlass.Int32(0)))"
            ),
            statement_from_string(
                f"{grouped_store_m} = min(cutlass.Int32({source_tile_m}), "
                f"{metadata_load_expr(3)} - {tile_start_var})"
            ),
        ]

    def _build_specialized_grouped_static_role_local_while(
        self,
        device_function: DeviceFunction,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        *,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None,
        role_prelude_stmts: list[ast.stmt] | None,
        initialize_tile_counter: bool,
    ) -> ast.stmt:
        """Emit a constant grouped schedule guarded by runtime metadata checks."""
        assert role_block.role_predicate is not None
        plan = self._tcgen05_plan()
        assert plan is not None and plan.grouped is not None
        grouped = plan.grouped
        shapes = grouped.static_problem_shapes
        assert shapes is not None
        assert grouped.real_groups is None
        assert plan.accumulator_view == "mn"
        assert plan.cluster_m == 1 and plan.cluster_n == 1
        assert plan.l2_swizzle_size == 1

        linear_idx = device_function.new_var(
            f"{scheduler_var_prefix}_static_linear_idx"
        )
        local_idx = device_function.new_var(f"{scheduler_var_prefix}_static_local_idx")
        prelude = [
            statement_from_string(
                f"{linear_idx} = cutlass.Int32(cute.arch.block_idx()[2])"
            ),
            statement_from_string(f"{local_idx} = cutlass.Int32(0)"),
        ]
        tile_counter_var, increment_tile_counter_per_tile = (
            self._finish_role_local_prelude(
                device_function,
                role_block,
                prelude,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
        )

        source_tile_m = plan.source_tile_m
        source_tile_n = plan.source_tile_n
        global_m_start = 0
        group_specs: list[tuple[int, int, int, int, int, int, int]] = []
        for metadata_idx, (problem_m, problem_n, problem_k) in enumerate(shapes):
            m_tiles = (problem_m + source_tile_m - 1) // source_tile_m
            n_tiles = (problem_n + source_tile_n - 1) // source_tile_n
            group_specs.append(
                (
                    metadata_idx,
                    problem_m,
                    problem_n,
                    problem_k,
                    m_tiles,
                    n_tiles,
                    global_m_start,
                )
            )
            global_m_start += m_tiles * source_tile_m

        tile_counts = [m_tiles * n_tiles for *_, m_tiles, n_tiles, _ in group_specs]
        # Keep every CTA on one group so dynamic TensorMaps are programmed only
        # once. The device-specific wrapper supplies host-computed quotas, so
        # each CTA only performs the group dispatch and tile-stride loop.
        quota_vars = list(grouped.static_group_quota_args)
        assert len(quota_vars) == len(group_specs)

        group_bodies: list[list[ast.stmt]] = []
        quota_prefix: list[str] = []
        for (
            metadata_idx,
            problem_m,
            problem_n,
            problem_k,
            m_tiles,
            _n_tiles,
            group_m_start,
        ), tile_count, quota in zip(group_specs, tile_counts, quota_vars, strict=True):
            block_offset = " + ".join(quota_prefix)
            local_expr = (
                linear_idx if not block_offset else f"{linear_idx} - ({block_offset})"
            )
            group_prelude = [
                f"{grouped.metadata_idx} = cutlass.Int32({metadata_idx})",
                f"{grouped.group_idx} = cutlass.Int32({metadata_idx})",
                f"{grouped.problem_m} = cutlass.Int32({problem_m})",
                f"{grouped.problem_n} = cutlass.Int32({problem_n})",
                f"{grouped.problem_k} = cutlass.Int32({problem_k})",
                f"{grouped.global_m_start} = cutlass.Int32({group_m_start})",
                f"{local_idx} = {local_expr}",
            ]
            per_tile_body = [
                statement_from_string(
                    f"{grouped.cta_tile_idx_m} = {local_idx} % cutlass.Int32({m_tiles})"
                ),
                statement_from_string(
                    f"{grouped.cta_tile_idx_n} = {local_idx} // "
                    f"cutlass.Int32({m_tiles})"
                ),
                statement_from_string(f"pid_0 = {grouped.cta_tile_idx_m}"),
                statement_from_string(f"pid_1 = {grouped.cta_tile_idx_n}"),
                statement_from_string(
                    f"tile_offset_0 = {grouped.global_m_start} + "
                    f"{grouped.cta_tile_idx_m} * cutlass.Int32({source_tile_m})"
                ),
                statement_from_string(
                    f"tile_offset_1 = {grouped.cta_tile_idx_n} * "
                    f"cutlass.Int32({source_tile_n})"
                ),
                *(
                    _clone_stmt(stmt)
                    for stmt in self._grouped_static_dependency_stmts(dependency_stmts)
                ),
                *(_clone_stmt(stmt) for stmt in role_block.stmts),
            ]
            if tile_counter_var is not None and increment_tile_counter_per_tile:
                per_tile_body.append(
                    statement_from_string(
                        f"{tile_counter_var} = {tile_counter_var} + cutlass.Int32(1)"
                    )
                )
            per_tile_body.append(
                statement_from_string(f"{local_idx} = {local_idx} + {quota}")
            )
            group_bodies.append(
                [
                    *(statement_from_string(line) for line in group_prelude),
                    create(
                        ast.While,
                        test=expr_from_string(
                            f"{local_idx} < cutlass.Int32({tile_count})"
                        ),
                        body=per_tile_body,
                        orelse=[],
                    ),
                ]
            )
            quota_prefix.append(quota)

        if len(group_bodies) == 1:
            prelude.extend(group_bodies[0])
        else:
            dispatch: list[ast.stmt] = group_bodies[-1]
            for index in range(len(group_bodies) - 2, -1, -1):
                quota_end = " + ".join(quota_vars[: index + 1])
                dispatch = [
                    create(
                        ast.If,
                        test=expr_from_string(f"{linear_idx} < {quota_end}"),
                        body=group_bodies[index],
                        orelse=dispatch,
                    )
                ]
            prelude.extend(dispatch)

        return create(
            ast.If,
            test=expr_from_string(role_block.role_predicate),
            body=prelude,
            orelse=[],
        )

    def _build_generic_grouped_static_role_local_while(
        self,
        device_function: DeviceFunction,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        *,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None,
        role_prelude_stmts: list[ast.stmt] | None = None,
        initialize_tile_counter: bool = True,
    ) -> ast.stmt:
        assert role_block.role_predicate is not None
        plan = self._tcgen05_plan()
        assert plan is not None and plan.grouped is not None
        grouped = plan.grouped
        assert plan.accumulator_view != "nm"

        sched_var = device_function.new_var(f"{scheduler_var_prefix}_grouped_sched")
        work_tile_var = device_function.new_var(f"{scheduler_var_prefix}_grouped_work")
        group_info_var = device_function.new_var(
            f"{scheduler_var_prefix}_group_search_result"
        )

        prelude: list[ast.stmt] = [
            statement_from_string(
                f"{sched_var} = cutlass.utils.StaticPersistentGroupTileScheduler.create("
                f"{grouped.sched_params}, cute.arch.block_idx(), "
                f"cute.arch.grid_dim(), ({plan.bm}, {plan.bn}, {plan.bk}), "
                "cutlass.utils.create_initial_search_state(), "
                f"{grouped.count}, {grouped.problem_sizes})"
            ),
            statement_from_string(
                f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
            ),
        ]

        tile_counter_var, increment_tile_counter_per_tile = (
            self._finish_role_local_prelude(
                device_function,
                role_block,
                prelude,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
        )

        grouped_cta_tile_idx_m = self._tcgen05_logical_m_coord_expr(
            f"{group_info_var}.cta_tile_idx_m"
        )
        grouped_cta_tile_idx_n = f"{group_info_var}.cta_tile_idx_n"
        grouped_metadata_stmts = [
            statement_from_string(
                f"{group_info_var} = {work_tile_var}.group_search_result"
            ),
            statement_from_string(
                f"{grouped.metadata_idx} = {group_info_var}.group_idx"
            ),
            *(
                [
                    statement_from_string(
                        f"{grouped.group_idx} = "
                        f"({grouped.real_groups}.iterator + "
                        f"{grouped.metadata_idx} * cutlass.Int32("
                        f"{grouped.real_groups}.layout.stride[0])).load()"
                    )
                ]
                if grouped.real_groups is not None
                else [
                    statement_from_string(
                        f"{grouped.group_idx} = {grouped.metadata_idx}"
                    )
                ]
            ),
            statement_from_string(
                f"{grouped.problem_m} = {group_info_var}.problem_shape_m"
            ),
            statement_from_string(
                f"{grouped.problem_n} = {group_info_var}.problem_shape_n"
            ),
            statement_from_string(
                f"{grouped.problem_k} = {group_info_var}.problem_shape_k"
            ),
            statement_from_string(
                f"{grouped.cta_tile_idx_m} = {grouped_cta_tile_idx_m}"
            ),
            statement_from_string(
                f"{grouped.cta_tile_idx_n} = {grouped_cta_tile_idx_n}"
            ),
            statement_from_string(
                f"{grouped.global_m_start} = "
                f"({grouped.starts}.iterator + {grouped.metadata_idx} "
                f"* cutlass.Int32({grouped.starts}.layout.stride[0])).load()"
            ),
            statement_from_string(f"pid_0 = {grouped.cta_tile_idx_m}"),
            statement_from_string(f"pid_1 = {grouped.cta_tile_idx_n}"),
            statement_from_string(
                f"tile_offset_0 = {grouped.global_m_start} + "
                f"{grouped.cta_tile_idx_m} * "
                f"cutlass.Int32({plan.source_tile_m})"
            ),
            statement_from_string(
                f"tile_offset_1 = {grouped.cta_tile_idx_n} * "
                f"cutlass.Int32({plan.source_tile_n})"
            ),
        ]
        per_tile_body: list[ast.stmt] = grouped_metadata_stmts
        per_tile_body.extend(self._grouped_static_dependency_stmts(dependency_stmts))
        per_tile_body.extend(role_block.stmts)
        if tile_counter_var is not None and increment_tile_counter_per_tile:
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

    def _build_grouped_static_role_local_while(
        self,
        device_function: DeviceFunction,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        *,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None,
        role_prelude_stmts: list[ast.stmt] | None = None,
        initialize_tile_counter: bool = True,
    ) -> ast.stmt:
        plan = self._tcgen05_plan()
        assert plan is not None and plan.grouped is not None
        shapes = plan.grouped.static_problem_shapes
        if (
            shapes is None
            or len(shapes) > TCGEN05_GROUPED_STATIC_SPECIALIZATION_MAX_GROUPS
        ):
            return self._build_generic_grouped_static_role_local_while(
                device_function,
                role_block,
                scheduler_var_prefix=scheduler_var_prefix,
                dependency_stmts=dependency_stmts,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )

        specialized = self._build_specialized_grouped_static_role_local_while(
            device_function,
            role_block,
            scheduler_var_prefix=scheduler_var_prefix,
            dependency_stmts=dependency_stmts,
            role_prelude_stmts=role_prelude_stmts,
            initialize_tile_counter=initialize_tile_counter,
        )
        generic = self._build_generic_grouped_static_role_local_while(
            device_function,
            role_block,
            scheduler_var_prefix=scheduler_var_prefix,
            dependency_stmts=dependency_stmts,
            role_prelude_stmts=(
                [_clone_stmt(stmt) for stmt in role_prelude_stmts]
                if role_prelude_stmts is not None
                else None
            ),
            initialize_tile_counter=initialize_tile_counter,
        )
        return create(
            ast.If,
            test=expr_from_string(
                f"cute.arch.grid_dim()[2] >= cutlass.Int32({len(shapes)})"
            ),
            body=[specialized],
            orelse=[generic],
        )

    def _build_grouped_static_role_local_while_with_scheduler(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        *,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None,
        role_prelude_stmts: list[ast.stmt] | None = None,
        emit_pdl_wait: bool = True,
        initialize_tile_counter: bool = True,
    ) -> ast.stmt:
        assert role_block.role_predicate is not None
        assert self._tcgen05_uses_grouped_worklist_nm_scheduler_mailbox()
        plan = self._tcgen05_plan()
        assert plan is not None
        grouped = plan.grouped
        assert grouped is not None
        sched_pipeline_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_pipeline_plan is not None

        sched_pipeline = sched_pipeline_plan.pipeline
        sched_consumer_state = sched_pipeline_plan.consumer_state
        valid_var = device_function.new_var(f"{scheduler_var_prefix}_valid")

        prelude: list[ast.stmt] = []
        if (
            emit_pdl_wait
            and self._tcgen05_is_two_cta()
            and role_block.role_predicate == self._tcgen05_tma_load_role_predicate()
        ):
            prelude.append(statement_from_string("cute.arch.griddepcontrol_wait()"))

        tile_counter_var, increment_tile_counter_per_tile = (
            self._finish_role_local_prelude(
                device_function,
                role_block,
                prelude,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
        )

        work_tile_stage_index = (
            f"{sched_consumer_state}.index"
            if self._tcgen05_uses_staged_work_tile_mailbox()
            else None
        )

        def slot(i: int) -> str:
            return self._tcgen05_work_tile_slot(layout, i)

        def _consumer_wait_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_wait_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
                work_tile_smem=layout.work_tile_smem,
                valid_var=valid_var,
                valid_slot_index=_TCGEN05_GROUPED_SELECTED_MAILBOX_VALID,
                work_tile_stage_index=work_tile_stage_index,
            )

        def _consumer_release_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_release_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
            )

        grouped_valid_store_m_stmts = (
            self._grouped_worklist_nm_valid_store_m_stmts(device_function)
            if role_block.role_predicate == self._tcgen05_epi_role_predicate()
            else []
        )
        mailbox_fields = (
            (grouped.cta_tile_idx_m, _TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_M),
            (grouped.cta_tile_idx_n, _TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_N),
            (grouped.metadata_idx, _TCGEN05_GROUPED_SELECTED_MAILBOX_METADATA_IDX),
            (grouped.group_idx, _TCGEN05_GROUPED_SELECTED_MAILBOX_GROUP_IDX),
            (grouped.problem_m, _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_M),
            (grouped.problem_n, _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_N),
            (grouped.problem_k, _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_K),
            (
                grouped.global_m_start,
                _TCGEN05_GROUPED_SELECTED_MAILBOX_GLOBAL_M_START,
            ),
        )
        grouped_metadata_read_stmts = [
            statement_from_string(f"{name} = {slot(field)}")
            for name, field in mailbox_fields
        ]
        grouped_metadata_read_stmts.extend(grouped_valid_store_m_stmts)
        grouped_metadata_read_stmts.extend(
            [
                statement_from_string(f"pid_0 = {grouped.cta_tile_idx_m}"),
                statement_from_string(f"pid_1 = {grouped.cta_tile_idx_n}"),
            ]
        )

        prelude.extend(_consumer_wait_block())
        per_tile_body: list[ast.stmt] = grouped_metadata_read_stmts
        per_tile_body.extend(_consumer_release_block())
        per_tile_body.extend(self._grouped_static_dependency_stmts(dependency_stmts))
        per_tile_body.extend(role_block.stmts)
        if tile_counter_var is not None and increment_tile_counter_per_tile:
            per_tile_body.append(
                statement_from_string(
                    f"{tile_counter_var} = {tile_counter_var} + cutlass.Int32(1)"
                )
            )
        per_tile_body.extend(_consumer_wait_block())
        prelude.append(
            create(
                ast.While,
                test=expr_from_string(valid_var),
                body=per_tile_body,
                orelse=[],
            )
        )
        prelude.extend(_consumer_release_block())
        return create(
            ast.If,
            test=expr_from_string(role_block.role_predicate),
            body=prelude,
            orelse=[],
        )

    def _build_role_local_while_with_scheduler(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
        role_block: Tcgen05PersistentProgramIDs._PersistentRoleBlock,
        *,
        scheduler_var_prefix: str,
        dependency_stmts: list[ast.stmt] | None,
        role_prelude_stmts: list[ast.stmt] | None = None,
        emit_pdl_wait: bool = True,
        initialize_tile_counter: bool = True,
        store_aux_per_tile_stmts: list[ast.stmt] | None = None,
        store_aux_predicate: str | None = None,
    ) -> ast.stmt:
        """``ROLE_LOCAL_WITH_SCHEDULER`` consumer-side role-local while.

        Each consumer role waits on the sched_pipeline, reads the
        published tile metadata from ``layout.work_tile_smem``, releases the
        sched stage, then runs its role block. The scheduler-warp role
        (built by ``_build_scheduler_warp_role_local_while``) owns
        the producer side: it runs ``StaticPersistentTileScheduler``
        and publishes per-tile metadata into the same SMEM mailbox.

        Cross-role producer-consumer synchronization for the *AB*
        and *acc* pipelines stays unchanged — those barriers carry
        the operand / accumulator data dependencies between the
        TMA-load, MMA-exec, and epi roles. The sched_pipeline is
        *only* used to broadcast per-tile coordinates.
        """
        assert role_block.role_predicate is not None
        sched_pipeline_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_pipeline_plan is not None, (
            "ROLE_LOCAL_WITH_SCHEDULER requires a registered "
            "sched_pipeline plan; was cute_state.register_tcgen05_sched_pipeline_plan "
            "called by _codegen_cute_mma?"
        )

        plan = self._tcgen05_plan()
        assert plan is not None and plan.has_scheduler_warp

        # Local handles for sched_pipeline variable names.
        sched_pipeline = sched_pipeline_plan.pipeline
        sched_consumer_state = sched_pipeline_plan.consumer_state
        # Per-role variable for the linearized virtual pid read out
        # of the SMEM mailbox each iteration.
        valid_var = device_function.new_var(f"{scheduler_var_prefix}_valid")

        prelude: list[ast.stmt] = []
        if (
            emit_pdl_wait
            and self._tcgen05_is_two_cta()
            and role_block.role_predicate == self._tcgen05_tma_load_role_predicate()
        ):
            # Same PDL hand-off as the MONOLITHIC path.
            prelude.append(statement_from_string("cute.arch.griddepcontrol_wait()"))

        tile_counter_var, increment_tile_counter_per_tile = (
            self._finish_role_local_prelude(
                device_function,
                role_block,
                prelude,
                role_prelude_stmts=role_prelude_stmts,
                initialize_tile_counter=initialize_tile_counter,
            )
        )

        work_tile_stage_index = (
            f"{sched_consumer_state}.index"
            if self._tcgen05_uses_staged_work_tile_mailbox()
            else None
        )

        # Linear virtual pid expression: the scheduler warp publishes
        # work-tile coordinates into ``layout.work_tile_smem``; we
        # reconstruct the linear pid the same way the MONOLITHIC path
        # does, just sourcing from SMEM coords instead of the
        # work_tile object.
        coord_terms = [
            self._tcgen05_work_tile_slot(layout, i) for i in range(len(self.pid_info))
        ]
        linear_pid_expr = self._tcgen05_linear_virtual_pid_from_coords_expr(coord_terms)

        # ``PipelineAsync.consumer_wait`` and ``consumer_release``
        # use the shared sched-pipeline factories at module scope —
        # see ``_build_sched_pipeline_consumer_wait_block`` and
        # ``_build_sched_pipeline_consumer_release_block`` for the
        # per-thread vs per-warp arrival count rationale. The
        # closures here just thread the per-role variables (the
        # role's ``valid_var`` plus the shared pipeline/state) into
        # fresh AST nodes per insertion point.
        def _consumer_wait_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_wait_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
                work_tile_smem=layout.work_tile_smem,
                valid_var=valid_var,
                work_tile_stage_index=work_tile_stage_index,
            )

        def _consumer_release_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_release_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
            )

        # Initial wait + valid-flag read happens in the prelude so the
        # ``while`` test can be a simple condition (CuTe DSL forbids
        # ``break`` inside ``@cute.kernel``).
        prelude.extend(_consumer_wait_block())
        per_tile_body: list[ast.stmt] = [
            statement_from_string(f"{self.virtual_pid_var} = {linear_pid_expr}"),
        ]
        # Match Quack's TileScheduler::get_current_work ordering: after the
        # role reads the published tile metadata, release the scheduler stage
        # immediately so the scheduler warp can publish the next tile while this
        # role processes the current tile.
        per_tile_body.extend(_consumer_release_block())
        if dependency_stmts is not None:
            per_tile_body.extend(dependency_stmts)
        # Cycle-94 merge: inject the store warp's aux GMEM->SMEM residual
        # producer body AFTER the tile coords are materialized and BEFORE the
        # store body (whose epi-warp consumers read the freshly staged aux),
        # gated on the store-warp predicate so only the single producer warp
        # issues the loads. The epi loop already owns the per-warp sched
        # handshake, so the injected body carries no sched wait/release.
        if store_aux_per_tile_stmts is not None:
            assert store_aux_predicate is not None
            per_tile_body.append(
                create(
                    ast.If,
                    test=expr_from_string(store_aux_predicate),
                    body=[_clone_stmt(stmt) for stmt in store_aux_per_tile_stmts],
                    orelse=[],
                )
            )
        per_tile_body.extend(role_block.stmts)
        if tile_counter_var is not None and increment_tile_counter_per_tile:
            per_tile_body.append(
                statement_from_string(
                    f"{tile_counter_var} = {tile_counter_var} + cutlass.Int32(1)"
                )
            )
        per_tile_body.extend(_consumer_wait_block())
        prelude.append(
            create(
                ast.While,
                test=expr_from_string(valid_var),
                body=per_tile_body,
                orelse=[],
            )
        )
        # Final release + advance for the sentinel publish (lane-0
        # gate matches the per-iteration release inside the loop).
        prelude.extend(_consumer_release_block())
        # Cycle-94 merge: no post-loop aux producer tail is injected. The store
        # warp's aux producer_state advance lives inside the per-tile store-warp
        # branch of THIS while; a post-loop ``producer_tail(state)`` would
        # reference a loop-carried value defined in that nested region (IR
        # domination error). The boundary drain is unnecessary because the
        # epi-warp consumers release every aux stage by loop exit (see
        # ``_build_c_input_warp_role_local_while(inline_aux_only=...)``).

        return create(
            ast.If,
            test=expr_from_string(role_block.role_predicate),
            body=prelude,
            orelse=[],
        )

    def _build_grouped_worklist_nm_scheduler_warp_role_local_while(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
    ) -> ast.stmt:
        assert self._tcgen05_uses_grouped_worklist_nm_scheduler_mailbox()
        plan = self._tcgen05_plan()
        assert plan is not None and plan.grouped is not None
        grouped = plan.grouped
        source_tile_m = plan.source_tile_m
        source_tile_n = plan.source_tile_n
        grouped_starts = grouped.starts
        grouped_real_groups = grouped.real_groups

        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        sched_pipeline = sched_plan.pipeline
        sched_producer_state = sched_plan.producer_state

        sched_var = device_function.new_var("tcgen05_grouped_selected_sched")
        work_tile_var = device_function.new_var("tcgen05_grouped_selected_work_tile")
        group_info_var = device_function.new_var(
            "tcgen05_grouped_selected_group_search_result"
        )
        metadata_idx_var = device_function.new_var(
            "tcgen05_grouped_selected_metadata_idx"
        )
        group_idx_var = device_function.new_var("tcgen05_grouped_selected_group_idx")
        cta_tile_idx_m_var = device_function.new_var(
            "tcgen05_grouped_selected_cta_tile_idx_m"
        )
        cta_tile_idx_n_var = device_function.new_var(
            "tcgen05_grouped_selected_cta_tile_idx_n"
        )
        source_n_tiles_var = device_function.new_var(
            "tcgen05_grouped_selected_source_n_tiles"
        )
        source_m_tiles_var = device_function.new_var(
            "tcgen05_grouped_selected_source_m_tiles"
        )
        source_m_fast_linear_var = device_function.new_var(
            "tcgen05_grouped_selected_source_m_fast_linear"
        )

        leader_predicate = "cute.arch.lane_idx() == cutlass.Int32(0)"

        def slot(i: int) -> str:
            return self._tcgen05_work_tile_producer_slot(layout, i)

        mailbox_values = (
            (_TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_M, cta_tile_idx_m_var),
            (_TCGEN05_GROUPED_SELECTED_MAILBOX_CTA_N, cta_tile_idx_n_var),
            (_TCGEN05_GROUPED_SELECTED_MAILBOX_VALID, "cutlass.Int32(1)"),
            (_TCGEN05_GROUPED_SELECTED_MAILBOX_METADATA_IDX, metadata_idx_var),
            (_TCGEN05_GROUPED_SELECTED_MAILBOX_GROUP_IDX, group_idx_var),
            (
                _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_M,
                f"{group_info_var}.problem_shape_n",
            ),
            (
                _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_N,
                f"{group_info_var}.problem_shape_m",
            ),
            (
                _TCGEN05_GROUPED_SELECTED_MAILBOX_PROBLEM_K,
                f"{group_info_var}.problem_shape_k",
            ),
            (
                _TCGEN05_GROUPED_SELECTED_MAILBOX_GLOBAL_M_START,
                (
                    f"({grouped_starts}.iterator + {metadata_idx_var} * "
                    f"cutlass.Int32({grouped_starts}.layout.stride[0])).load()"
                ),
            ),
        )

        def source_m_fast_raster_stmts() -> list[ast.stmt]:
            backend = CompileEnvironment.current().backend
            source_n_tiles_expr = backend.cdiv_expr(
                f"{group_info_var}.problem_shape_m",
                f"cutlass.Int32({source_tile_n})",
                is_device=True,
            )
            source_m_tiles_expr = (
                backend.cdiv_expr(
                    f"{group_info_var}.problem_shape_n",
                    f"cutlass.Int32({source_tile_m})",
                    is_device=True,
                )
                if grouped.device_split_sizes
                else (
                    f"{group_info_var}.problem_shape_n // "
                    f"cutlass.Int32({source_tile_m})"
                )
            )
            return [
                statement_from_string(f"{source_n_tiles_var} = {source_n_tiles_expr}"),
                statement_from_string(f"{source_m_tiles_var} = {source_m_tiles_expr}"),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{source_m_tiles_var} <= {source_n_tiles_var}"
                    ),
                    body=[
                        # For wide compact rows, walk source-M fastest to keep
                        # adjacent CTAs on nearby packed-A rows.
                        statement_from_string(
                            f"{source_m_fast_linear_var} = "
                            f"{cta_tile_idx_m_var} * {source_n_tiles_var} + "
                            f"{cta_tile_idx_n_var}"
                        ),
                        statement_from_string(
                            f"{cta_tile_idx_m_var} = "
                            f"{source_m_fast_linear_var} % {source_m_tiles_var}"
                        ),
                        statement_from_string(
                            f"{cta_tile_idx_n_var} = "
                            f"{source_m_fast_linear_var} // {source_m_tiles_var}"
                        ),
                    ],
                    orelse=[],
                ),
            ]

        def publish_current_tile_leader_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                statement_from_string(
                    f"{group_info_var} = {work_tile_var}.group_search_result"
                ),
                statement_from_string(
                    f"{metadata_idx_var} = {group_info_var}.group_idx"
                ),
                *(
                    [
                        statement_from_string(
                            f"{group_idx_var} = "
                            f"({grouped_real_groups}.iterator + "
                            f"{metadata_idx_var} * cutlass.Int32("
                            f"{grouped_real_groups}.layout.stride[0])).load()"
                        )
                    ]
                    if grouped_real_groups
                    else [
                        statement_from_string(f"{group_idx_var} = {metadata_idx_var}")
                    ]
                ),
                statement_from_string(
                    f"{cta_tile_idx_m_var} = {group_info_var}.cta_tile_idx_n"
                ),
                statement_from_string(
                    f"{cta_tile_idx_n_var} = "
                    f"{self._tcgen05_logical_m_coord_expr(f'{group_info_var}.cta_tile_idx_m')}"
                ),
                *source_m_fast_raster_stmts(),
                *[
                    statement_from_string(f"{slot(field)} = {value}")
                    for field, value in mailbox_values
                ],
                statement_from_string(
                    f"{sched_pipeline}.producer_commit({sched_producer_state})"
                ),
            ]

        def publish_current_tile_stmts() -> list[ast.stmt]:
            return [
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=publish_current_tile_leader_stmts(),
                    orelse=[],
                ),
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
                statement_from_string("cute.arch.sync_warp()"),
            ]

        def scheduler_advance_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(f"{sched_var}.advance_to_next_work()"),
                statement_from_string(
                    f"{work_tile_var} = {sched_var}.get_current_work()"
                ),
            ]

        prelude: list[ast.stmt] = [
            statement_from_string(
                f"{sched_var} = cutlass.utils.StaticPersistentGroupTileScheduler.create("
                f"{grouped.sched_params}, cute.arch.block_idx(), "
                f"cute.arch.grid_dim(), ({plan.bm}, {plan.bn}, {plan.bk}), "
                "cutlass.utils.create_initial_search_state(), "
                f"{grouped.count}, {grouped.problem_sizes})"
            ),
            statement_from_string(
                f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
            ),
            create(
                ast.While,
                test=expr_from_string(f"{work_tile_var}.is_valid_tile"),
                body=[*publish_current_tile_stmts(), *scheduler_advance_stmts()],
                orelse=[],
            ),
        ]

        prelude.extend(
            [
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=[
                        statement_from_string(
                            f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                        ),
                        statement_from_string(
                            f"{slot(_TCGEN05_GROUPED_SELECTED_MAILBOX_VALID)} = "
                            "cutlass.Int32(0)"
                        ),
                        statement_from_string(
                            f"{sched_pipeline}.producer_commit({sched_producer_state})"
                        ),
                    ],
                    orelse=[],
                ),
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
                statement_from_string("cute.arch.sync_warp()"),
            ]
        )
        return create(
            ast.If,
            test=expr_from_string(self._tcgen05_scheduler_role_predicate()),
            body=prelude,
            orelse=[],
        )

    def _build_scheduler_warp_role_local_while(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
    ) -> ast.stmt:
        """Build the scheduler-warp's role-local while.

        Active only under ``ROLE_LOCAL_WITH_SCHEDULER``. The scheduler
        warp owns the persistent tile scheduler and publishes
        per-tile metadata + a sentinel via the broadcast pipeline.
        Consumer warps (TMA-load, MMA-exec, epi) wait on the same
        pipeline and read from ``layout.work_tile_smem``.

        The body is constructed *here* rather than extracted from
        device IR because the scheduler-warp work has no source
        statements in the user kernel — it is pure scheduling
        infrastructure.

        Dispatches by persistence model:

        - ``STATIC_PERSISTENT`` (default): emits
          ``StaticPersistentTileScheduler.create`` + the static
          persistent loop.
        - ``CLC_PERSISTENT`` (G2-H, cute_plan.md): emits
          ``nvvm.clusterlaunchcontrol_try_cancel`` per persistent-loop
          iteration; the response decoder unpacks the next cluster's
          CTA id (or a "canceled" sentinel) into the SMEM mailbox the
          consumer warps already read from. Active only on arch >= 100
          per ``validate_tcgen05_strategy_invariants``.
        """
        plan = self._tcgen05_plan()
        assert plan is not None and plan.has_scheduler_warp
        if self._tcgen05_uses_grouped_worklist_nm_scheduler_mailbox():
            return self._build_grouped_worklist_nm_scheduler_warp_role_local_while(
                device_function, layout
            )
        if plan.is_clc_persistent:
            return self._build_scheduler_warp_role_local_while_clc(
                device_function, layout
            )
        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        sched_pipeline = sched_plan.pipeline
        sched_producer_state = sched_plan.producer_state

        sched_params_var = device_function.new_var(
            "tcgen05_scheduler_warp_tile_sched_params"
        )
        sched_var = device_function.new_var("tcgen05_scheduler_warp_tile_sched")
        work_tile_var = device_function.new_var("tcgen05_scheduler_warp_work_tile")

        # ``PipelineAsync`` producer/consumer mbarrier ops are
        # per-thread; with ``producer_arrive_count = 1`` only one
        # thread should arrive on the full barrier per stage.
        # Gate the producer ops + SMEM writes on lane 0 of the
        # scheduler warp. ``mbarrier.wait`` (used by
        # ``producer_acquire``) is fine to call from any thread —
        # PTX semantics stall the thread until the phase flips —
        # but I keep the leader-only gate for it too, paired with
        # a ``sync_warp`` after, so the SMEM write order vs the
        # warp's other 31 lanes is well-defined. The 31 non-leader
        # lanes do nothing per iteration; ``advance_to_next_work``
        # mutates register-resident state that all 32 threads
        # share via warp-uniform reads.
        leader_predicate = "cute.arch.lane_idx() == cutlass.Int32(0)"

        def publish_current_tile_leader_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(0)] = {work_tile_var}.tile_idx[0]"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(1)] = {work_tile_var}.tile_idx[1]"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(2)] = {work_tile_var}.tile_idx[2]"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(3)] = "
                    f"(cutlass.Int32(1) if {work_tile_var}.is_valid_tile "
                    f"else cutlass.Int32(0))"
                ),
                statement_from_string(
                    f"{sched_pipeline}.producer_commit({sched_producer_state})"
                ),
            ]

        def publish_current_tile_stmts() -> list[ast.stmt]:
            return [
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=publish_current_tile_leader_stmts(),
                    orelse=[],
                ),
                # Advance state on every lane so all 32 threads stay in
                # sync on the producer state register. Then sync_warp
                # so the leader's SMEM writes are observable to lanes
                # 1-31 (defensive — they don't read this SMEM, but it
                # keeps the warp's view of memory uniform for any
                # future reads).
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
                statement_from_string("cute.arch.sync_warp()"),
            ]

        def scheduler_advance_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(f"{sched_var}.advance_to_next_work()"),
                statement_from_string(
                    f"{work_tile_var} = {sched_var}.get_current_work()"
                ),
            ]

        def per_tile_body(*, publish_if: str | None = None) -> list[ast.stmt]:
            if publish_if is None:
                return [
                    *publish_current_tile_stmts(),
                    *scheduler_advance_stmts(),
                ]
            return [
                create(
                    ast.If,
                    test=expr_from_string(publish_if),
                    body=publish_current_tile_stmts(),
                    orelse=[],
                ),
                *scheduler_advance_stmts(),
            ]

        # Producer loop: while the current work tile is valid, publish
        # it and advance to the next. The final sentinel publish (with
        # ``is_valid=False``) happens *outside* the loop so the
        # consumer warps see exactly one trailing invalid arrival
        # after the last valid tile. CuTe DSL forbids ``break`` inside
        # ``@cute.kernel`` so the loop test runs on the
        # freshly-fetched ``work_tile_var``.
        prelude: list[ast.stmt] = [
            statement_from_string(
                f"{sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                f"{self._tcgen05_persistent_tile_sched_params_args(cluster_m=layout.cluster_m, cluster_n=layout.cluster_n)})"
            ),
        ]

        def scheduler_create_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(
                    f"{sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                    f"{sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
                ),
                statement_from_string(
                    f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
                ),
            ]

        prelude.extend(scheduler_create_stmts())

        # Sentinel publish after the loop exits: producer-only writes
        # gated on lane 0, then the producer arrive on the full
        # barrier (so the last consumer iteration sees an invalid
        # tile and exits the consumer loop).
        def sentinel_leader_stmts() -> list[ast.stmt]:
            return [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(0)] = cutlass.Int32(0)"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(1)] = cutlass.Int32(0)"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(2)] = cutlass.Int32(0)"
                ),
                statement_from_string(
                    f"{layout.work_tile_smem}[cutlass.Int32(3)] = cutlass.Int32(0)"
                ),
                statement_from_string(
                    f"{sched_pipeline}.producer_commit({sched_producer_state})"
                ),
            ]

        def sentinel_publish_stmts() -> list[ast.stmt]:
            return [
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=sentinel_leader_stmts(),
                    orelse=[],
                ),
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
                statement_from_string("cute.arch.sync_warp()"),
            ]

        full_edge_split = (
            plan.tma_store_full_tiles_only
            and device_function.cute_state.has_tcgen05_epi_role_full_edge_split
        )
        if full_edge_split:
            full_tile_var = device_function.new_var("tcgen05_scheduler_warp_full_tile")
            full_tile_expr = self._tcgen05_output_full_tile_expr_for_work_tile(
                work_tile_var
            )
            # The split scheduler scans the same static tile space twice:
            # first publishing interior full tiles, then publishing fringe
            # edge tiles after a sentinel and scheduler reset.
            for is_full_phase in (True, False):
                publish_if = full_tile_var if is_full_phase else f"not {full_tile_var}"
                prelude.append(
                    create(
                        ast.While,
                        test=expr_from_string(f"{work_tile_var}.is_valid_tile"),
                        body=[
                            statement_from_string(
                                f"{full_tile_var} = {full_tile_expr}"
                            ),
                            *per_tile_body(publish_if=publish_if),
                        ],
                        orelse=[],
                    )
                )
                prelude.extend(sentinel_publish_stmts())
                if is_full_phase:
                    prelude.extend(scheduler_create_stmts())
        else:
            prelude.append(
                create(
                    ast.While,
                    test=expr_from_string(f"{work_tile_var}.is_valid_tile"),
                    body=per_tile_body(),
                    orelse=[],
                )
            )
            prelude.extend(sentinel_publish_stmts())

        return create(
            ast.If,
            test=expr_from_string(self._tcgen05_scheduler_role_predicate()),
            body=prelude,
            orelse=[],
        )

    def _build_scheduler_warp_role_local_while_clc(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
    ) -> ast.stmt:
        """G2-H: CLC-driven scheduler-warp body (cute_plan.md).

        Replaces the ``StaticPersistentTileScheduler.create`` +
        ``advance_to_next_work``/``get_current_work`` pattern with a
        ``nvvm.clusterlaunchcontrol_try_cancel`` query per
        persistent-loop iteration. The CLC instruction asynchronously
        writes a 4 × Int32 response to the SMEM buffer allocated by
        ``_emit_clc_smem_setup`` (in ``cute_mma._codegen_cute_mma``);
        each response slot decodes to ``(bidx, bidy, bidz, valid)``
        via ``cute.arch.clc_response``.

        Topology (mirrors Quack's ``_fetch_next_work_idx`` CLC branch
        in ``quack/quack/tile_scheduler.py``):

        - cluster_m == 1: each CTA is its own cluster, so each CTA's
          scheduler warp issues an independent ``nomulticast``
          ``try_cancel`` against its own local SMEM and publishes
          locally.
        - cluster_m > 1: only the cluster leader CTA's scheduler
          warp issues the CLC query (matches Quack's
          ``is_scheduler_warp = block_idx_in_cluster() == 0``).
          Non-leader CTAs receive the response indirectly because
          the leader broadcasts the resulting work tile to every
          peer CTA's SMEM mailbox via ``_cute_store_shared_remote_x4``.
          Each peer CTA's consumer warps still wait/release on the
          per-CTA ``sched_pipeline``, so the per-CTA empty-barrier
          arrival counts the static WITH_SCHEDULER path validates
          stay unchanged.

        cluster_m collapse: the publish writes the per-CTA M
        coordinate into each peer's mailbox by adding
        ``peer_cta_rank_in_cluster_m`` to the CLC response's
        ``bidx`` (which encodes the *first* CTA of the next
        cluster). The consumer's ``// cluster_m`` collapse in
        ``_tcgen05_logical_m_coord_expr`` then converts back to
        the cluster-level virtual_pid for tile distribution.
        """
        plan = self._tcgen05_plan()
        assert plan is not None and plan.has_scheduler_warp and plan.is_clc_persistent
        sched_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_plan is not None
        sched_pipeline = sched_plan.pipeline
        sched_producer_state = sched_plan.producer_state
        sched_consumer_state = sched_plan.consumer_state
        clc_response_smem_ptr = sched_plan.clc_response_smem_ptr
        clc_mbar_smem_ptr = sched_plan.clc_mbar_smem_ptr
        clc_mbar_phase = sched_plan.clc_mbar_phase
        assert clc_response_smem_ptr and clc_mbar_smem_ptr and clc_mbar_phase, (
            "CLC scheduler-warp body requires sched plan SMEM/mbarrier "
            "names; was _new_tcgen05_sched_pipeline_plan called with "
            "use_clc=True?"
        )

        # Per-iteration response decoded into named locals so the
        # publish writes are linear and easy to read in generated
        # code.
        bidx_var = device_function.new_var("tcgen05_clc_bidx")
        bidy_var = device_function.new_var("tcgen05_clc_bidy")
        bidz_var = device_function.new_var("tcgen05_clc_bidz")
        valid_var = device_function.new_var("tcgen05_clc_valid")
        # Initial work-tile coordinates come from the launcher's
        # ``block_idx()`` so the first iteration runs the cluster
        # the launcher placed this CTA in. Subsequent iterations
        # come from the CLC response, with the per-CTA offset added
        # back in the publish step.
        cluster_bidx_var = device_function.new_var("tcgen05_clc_cluster_bidx")
        cluster_bidy_var = device_function.new_var("tcgen05_clc_cluster_bidy")
        cluster_bidz_var = device_function.new_var("tcgen05_clc_cluster_bidz")

        leader_predicate = "cute.arch.lane_idx() == cutlass.Int32(0)"

        # CLC mbarrier init: ``mbarrier_init(addr, 1)`` arms the
        # barrier with arrival count 1 (only the CLC issuer arrives).
        # Followed by ``mbarrier_init_fence`` + ``sync_warp`` per
        # Quack's ``_init_clc_mbarrier`` pattern. Gate on lane 0 so
        # only one thread runs the init op; the fence/sync make the
        # init visible to the other 31 lanes.
        clc_init_block: list[ast.stmt] = [
            create(
                ast.If,
                test=expr_from_string(leader_predicate),
                body=[
                    statement_from_string(
                        f"cute.arch.mbarrier_init({clc_mbar_smem_ptr}, 1)"
                    ),
                ],
                orelse=[],
            ),
            statement_from_string("cute.arch.mbarrier_init_fence()"),
            statement_from_string("cute.arch.sync_warp()"),
        ]

        # Initial cluster coordinates: decode block_idx[2] -> tile_idx
        # via ``StaticPersistentTileScheduler.create``. This matches
        # the persistent-grid encoding the launch grid uses
        # (``(cluster_m, 1, num_clusters)``) — block_idx[0] is the
        # CTA-in-cluster offset and block_idx[2] is the linear
        # cluster id, so the static scheduler's
        # ``_get_current_work_for_linear_idx`` is the right decoder.
        # CLC's response also returns CTAIDs in this coordinate
        # system (``bidz`` is the cluster id, ``bidx`` is the CTA
        # within cluster), so we can use the same decoder for
        # subsequent CLC responses by setting up a fresh
        # ``StaticPersistentTileScheduler`` from the CLC bidx/bidy/bidz.
        #
        # ``valid_var`` is Int32 because ``cute.arch.clc_response``
        # returns Int32 for the valid flag. The CuTe DSL's while-region
        # type-checker rejects type changes between iterations.
        sched_params_var = device_function.new_var("tcgen05_clc_initial_sched_params")
        sched_var = device_function.new_var("tcgen05_clc_initial_sched")
        work_tile_var = device_function.new_var("tcgen05_clc_initial_work_tile")
        clc_initial_block = [
            # Build the persistent tile-scheduler params for the
            # initial decode. ``layout.cluster_m/n`` agrees with the
            # launch grid's cluster shape ``(cluster_m, cluster_n, 1)``.
            statement_from_string(
                f"{sched_params_var} = cutlass.utils.PersistentTileSchedulerParams("
                f"{self._tcgen05_persistent_tile_sched_params_args(cluster_m=layout.cluster_m, cluster_n=layout.cluster_n)})"
            ),
            statement_from_string(
                f"{sched_var} = cutlass.utils.StaticPersistentTileScheduler.create("
                f"{sched_params_var}, cute.arch.block_idx(), cute.arch.grid_dim())"
            ),
            statement_from_string(
                f"{work_tile_var} = {sched_var}.initial_work_tile_info()"
            ),
            # Bind the initial cluster coords from the static
            # scheduler's decode. ``tile_idx[0]`` is already the
            # per-CTA M coordinate (= cluster_id_m * cluster_m +
            # cta_in_cluster_m) since the static scheduler folds the
            # cta_in_cluster offset in via
            # ``_get_current_work_for_linear_idx``.
            statement_from_string(f"{cluster_bidx_var} = {work_tile_var}.tile_idx[0]"),
            statement_from_string(f"{cluster_bidy_var} = {work_tile_var}.tile_idx[1]"),
            statement_from_string(f"{cluster_bidz_var} = {work_tile_var}.tile_idx[2]"),
            # Initial valid flag: the scheduler warp only runs if
            # the launcher placed it on a valid cluster. The CLC
            # query handles invalidation for subsequent waves.
            statement_from_string(
                f"{valid_var} = cutlass.Int32(1) "
                f"if {work_tile_var}.is_valid_tile else cutlass.Int32(0)"
            ),
        ]

        # Per-tile publish: write (bidx, bidy, bidz, valid) into the
        # work-tile mailbox.
        #
        # cluster_m == 1: leader writes its own local mailbox; consumer
        # waits on local empty mbar (per-CTA pipeline).
        # cluster_m  > 1: leader broadcasts to every peer CTA's mailbox
        # via ``_cute_store_shared_remote_x4`` and arms each peer's
        # full mbar with ``mbarrier_arrive_and_expect_tx``. Consumer
        # arrivals are routed to leader's empty mbar via
        # ``consumer_mask=Int32(0)`` (the cluster-leader topology set
        # up in ``cute_mma._codegen_cute_mma`` for the CLC path).
        #
        # ``producer_acquire`` is **per-thread** (its underlying
        # ``mbarrier.wait`` PTX stalls each issuing thread until the
        # phase flips), so every lane of the scheduler warp must call
        # it; gating it under a ``if lane_idx == 0`` would only stall
        # lane 0 and let the other 31 lanes race ahead, breaking the
        # producer/consumer handshake. Mirrors Quack's
        # ``write_work_tile_to_smem`` in
        # ``quack/quack/tile_scheduler.py`` which calls
        # ``producer_acquire`` on the full warp before the per-lane
        # ``if lane_idx < cluster_size`` branch.
        # Pre-declare the cluster-broadcast variable names so pyrefly
        # sees them defined unconditionally; their usage stays inside
        # ``if layout.cluster_m > 1`` branches below.
        sched_barrier_ptr = ""
        sched_peer_rank = ""
        sched_peer_m = ""
        staged_work_tile_mailbox = self._tcgen05_uses_staged_work_tile_mailbox()
        producer_barrier_state = (
            sched_producer_state if staged_work_tile_mailbox else sched_consumer_state
        )
        producer_smem_ptr = self._tcgen05_work_tile_producer_smem_ptr(layout)
        if layout.cluster_m > 1:
            sched_barrier_ptr = device_function.new_var("tcgen05_clc_sched_barrier_ptr")
            sched_peer_rank = device_function.new_var("tcgen05_clc_sched_peer_rank")
            sched_peer_m = device_function.new_var("tcgen05_clc_sched_peer_m")
            # Whole-warp prelude: every lane runs ``producer_acquire``
            # (mbarrier wait) and computes the warp-uniform barrier
            # pointer + lane id. Lanes ``cluster_m..31`` no-op past
            # the per-peer broadcast branch.
            per_tile_publish_warp = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                # Remote stores arm the current full barrier, matching Quack's
                # PipelineState pairing and the clustered static mailbox bridge.
                statement_from_string(
                    f"{sched_barrier_ptr} = "
                    f"{sched_pipeline}.producer_get_barrier({producer_barrier_state})"
                ),
                statement_from_string(f"{sched_peer_rank} = cute.arch.lane_idx()"),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{sched_peer_rank} < cutlass.Int32({layout.cluster_m})"
                    ),
                    body=[
                        statement_from_string(f"{sched_peer_m} = {sched_peer_rank}"),
                        statement_from_string(
                            "cute.arch.mbarrier_arrive_and_expect_tx("
                            f"{sched_barrier_ptr}, 16, {sched_peer_rank})"
                        ),
                        statement_from_string(
                            f"_cute_store_shared_remote_x4("
                            f"{cluster_bidx_var} + {sched_peer_m}, "
                            f"{cluster_bidy_var}, "
                            f"{cluster_bidz_var}, "
                            f"{valid_var}, "
                            f"smem_ptr={producer_smem_ptr}, "
                            f"mbar_ptr={sched_barrier_ptr}, "
                            f"peer_cta_rank_in_cluster={sched_peer_rank})"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        else:
            # cluster_m == 1: lane-0-only local publish + commit. Still
            # call ``producer_acquire`` on every lane (the mbarrier wait
            # must stall the whole warp), then the leader gate writes +
            # commits.
            per_tile_publish_warp = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=[
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 0)} = "
                            f"{cluster_bidx_var}"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 1)} = "
                            f"{cluster_bidy_var}"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 2)} = "
                            f"{cluster_bidz_var}"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 3)} = "
                            f"{valid_var}"
                        ),
                        statement_from_string(
                            f"{sched_pipeline}.producer_commit({sched_producer_state})"
                        ),
                    ],
                    orelse=[],
                ),
            ]

        # CLC query block: arm + issue + wait + decode. Quack's
        # pattern: lane 0 of the leader CTA's scheduler warp issues
        # the query, all 32 lanes of that warp wait, then
        # ``cute.arch.clc_response`` reads back from SMEM. ``try_cancel``
        # cancels exactly one cluster per call, so issuance is gated
        # to leader CTA only for ``cluster_m > 1``.
        # Update cluster coordinate locals so the next iteration's
        # publish uses the just-decoded response.
        #
        # CLC returns the CTAID of the canceled cluster's first CTA
        # in (bidx, bidy, bidz). With Helion's launch grid
        # ``(cluster_m, 1, num_clusters)`` the first CTA of cluster N
        # has CTAID ``(0, 0, N)``, so ``bidz`` IS the cluster id.
        # Reuse the existing ``StaticPersistentTileScheduler`` to
        # decode that cluster id back to per-CTA tile coordinates by
        # writing ``_current_work_linear_idx`` and calling
        # ``get_current_work()``. This matches what the static path
        # would have computed for the same cluster id, so the
        # consumer's ``virtual_pid = work_tile_smem[0] // cluster_m``
        # collapse continues to work.
        next_work_tile_var = device_function.new_var("tcgen05_clc_next_work_tile")
        clc_helper_call = "_cute_issue_clc_query_nomulticast"
        clc_query_block = [
            statement_from_string("cute.arch.sync_warp()"),
            create(
                ast.If,
                test=expr_from_string(leader_predicate),
                body=[
                    statement_from_string(
                        f"cute.arch.mbarrier_arrive_and_expect_tx("
                        f"{clc_mbar_smem_ptr}, 16)"
                    ),
                    statement_from_string(
                        f"{clc_helper_call}("
                        f"{clc_mbar_smem_ptr}, {clc_response_smem_ptr})"
                    ),
                ],
                orelse=[],
            ),
            statement_from_string("cute.arch.sync_warp()"),
            statement_from_string(
                f"cute.arch.mbarrier_wait({clc_mbar_smem_ptr}, {clc_mbar_phase})"
            ),
            statement_from_string(
                f"{clc_mbar_phase} = {clc_mbar_phase} ^ cutlass.Int32(1)"
            ),
            statement_from_string(
                f"({bidx_var}, {bidy_var}, {bidz_var}, {valid_var}) = "
                f"cute.arch.clc_response({clc_response_smem_ptr})"
            ),
            statement_from_string("cute.arch.fence_view_async_shared()"),
            statement_from_string(f"{sched_var}._current_work_linear_idx = {bidz_var}"),
            statement_from_string(
                f"{next_work_tile_var} = {sched_var}.get_current_work()"
            ),
            statement_from_string(
                f"{cluster_bidx_var} = {next_work_tile_var}.tile_idx[0]"
            ),
            statement_from_string(
                f"{cluster_bidy_var} = {next_work_tile_var}.tile_idx[1]"
            ),
            statement_from_string(
                f"{cluster_bidz_var} = {next_work_tile_var}.tile_idx[2]"
            ),
        ]

        # ``per_tile_publish_warp`` already does its own per-lane
        # gating internally (lane-0-only commit for cluster_m=1, the
        # ``lane_idx < cluster_m`` per-peer broadcast for cluster_m>1).
        # Inserting another leader-only ``if`` around it would gate
        # the entire publish to lane 0 and either skip the broadcast
        # to peer CTAs or stall only lane 0 on ``producer_acquire``.
        per_tile_body = [
            *per_tile_publish_warp,
            statement_from_string(emit_pipeline_advance(sched_producer_state)),
            *clc_query_block,
            statement_from_string("cute.arch.sync_warp()"),
        ]

        prelude: list[ast.stmt] = []
        # PDL (programmatic dependent launch) hand-off: the scheduler
        # warp must wait for the prior kernel before reading CLC state.
        # ``cute.arch.clc_response`` returns ``valid=0`` if the
        # ``griddepcontrol`` chain isn't established. Quack calls this
        # in the scheduler-warp body (see
        # ``quack/quack/gemm_sm100.py``: ``if const_expr(self.use_pdl):
        # cute.arch.griddepcontrol_wait()`` inside
        # ``if warp_idx == self.scheduler_warp_id:``). Without this the
        # CLC query reliably returns invalid on the very first call,
        # so the persistent loop terminates after iteration 0 and the
        # kernel produces only the initial-tile output.
        prelude.append(statement_from_string("cute.arch.griddepcontrol_wait()"))
        prelude.extend(clc_init_block)
        prelude.extend(clc_initial_block)
        # Producer loop: publish the current tile and then query the next one.
        # CuTe DSL forbids ``break`` so the loop test is on the
        # dynamically-updated ``valid_var``.
        # ``valid_var`` is Int32 because ``cute.arch.clc_response``
        # returns Int32 for the valid flag — comparing against 0
        # keeps the test type-stable across iterations.
        prelude.append(
            create(
                ast.While,
                test=expr_from_string(f"{valid_var} != cutlass.Int32(0)"),
                body=per_tile_body,
                orelse=[],
            )
        )
        # Sentinel publish after the loop exits so the consumer warps'
        # last-iteration wait sees an invalid tile and exits. The
        # sentinel mirrors the in-loop publish exactly: cluster_m>1
        # broadcasts ``(0, 0, 0, valid=0)`` to every peer CTA via
        # ``_cute_store_shared_remote_x4``; cluster_m=1 writes
        # locally. ``producer_acquire`` runs on every lane.
        if layout.cluster_m > 1:
            sentinel_warp: list[ast.stmt] = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                # Remote stores arm the current full barrier, matching Quack's
                # PipelineState pairing and the clustered static mailbox bridge.
                statement_from_string(
                    f"{sched_barrier_ptr} = "
                    f"{sched_pipeline}.producer_get_barrier({producer_barrier_state})"
                ),
                statement_from_string(f"{sched_peer_rank} = cute.arch.lane_idx()"),
                create(
                    ast.If,
                    test=expr_from_string(
                        f"{sched_peer_rank} < cutlass.Int32({layout.cluster_m})"
                    ),
                    body=[
                        statement_from_string(
                            "cute.arch.mbarrier_arrive_and_expect_tx("
                            f"{sched_barrier_ptr}, 16, {sched_peer_rank})"
                        ),
                        statement_from_string(
                            f"_cute_store_shared_remote_x4("
                            "cutlass.Int32(0), cutlass.Int32(0), "
                            "cutlass.Int32(0), cutlass.Int32(0), "
                            f"smem_ptr={producer_smem_ptr}, "
                            f"mbar_ptr={sched_barrier_ptr}, "
                            f"peer_cta_rank_in_cluster={sched_peer_rank})"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        else:
            sentinel_warp = [
                statement_from_string(
                    f"{sched_pipeline}.producer_acquire({sched_producer_state})"
                ),
                create(
                    ast.If,
                    test=expr_from_string(leader_predicate),
                    body=[
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 0)} = "
                            "cutlass.Int32(0)"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 1)} = "
                            "cutlass.Int32(0)"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 2)} = "
                            "cutlass.Int32(0)"
                        ),
                        statement_from_string(
                            f"{self._tcgen05_work_tile_producer_slot(layout, 3)} = "
                            "cutlass.Int32(0)"
                        ),
                        statement_from_string(
                            f"{sched_pipeline}.producer_commit({sched_producer_state})"
                        ),
                    ],
                    orelse=[],
                ),
            ]
        prelude.extend(
            [
                *sentinel_warp,
                statement_from_string(emit_pipeline_advance(sched_producer_state)),
                # Quack drains the scheduler pipeline from the whole scheduler
                # warp after publishing the invalid work tile. Helion's
                # scheduler warp does not consume its own mailbox, so tail here
                # waits for consumer roles to release the sentinel before the
                # scheduler role exits.
                statement_from_string(
                    f"{sched_pipeline}.producer_tail({sched_producer_state})"
                ),
                statement_from_string("cute.arch.sync_warp()"),
            ]
        )

        # cluster_m>1: gate the CLC body to leader CTA only (Quack
        # pattern). Non-leader CTAs' scheduler warps idle while the
        # leader broadcasts to every peer's mailbox via
        # ``_cute_store_shared_remote_x4``. The non-leader scheduler
        # warps' consumer-side wait/release is unaffected because the
        # leader's broadcast arms each peer's full mbar via cross-CTA
        # ``mbarrier_arrive_and_expect_tx``, and the leader's
        # ``producer_acquire`` waits on the cluster-routed empty mbar
        # (set up via ``consumer_mask_to_leader=True`` in
        # ``cute_mma._codegen_cute_mma`` for the CLC path).
        scheduler_predicate = self._tcgen05_scheduler_role_predicate()
        if layout.cluster_m > 1:
            scheduler_predicate = (
                f"({scheduler_predicate}) "
                "and cute.arch.make_warp_uniform("
                "cute.arch.block_idx_in_cluster()) == cutlass.Int32(0)"
            )
        return create(
            ast.If,
            test=expr_from_string(scheduler_predicate),
            body=prelude,
            orelse=[],
        )

    def _build_c_input_warp_role_local_while(
        self,
        device_function: DeviceFunction,
        layout: Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout,
        *,
        shared_body_extracted: list[ast.stmt] | None = None,
        tile_phase: str = "all",
        inline_aux_only: bool = False,
    ) -> ast.stmt | list[ast.stmt]:
        """Build the C-input warp's role-local while
        (``cute_plan.md`` §7.5.3.2 producer-body split).

        Workstream A Stage 5 (cycle 94, the merge): when ``inline_aux_only`` is
        set, this does NOT build a self-contained role-local while (which would
        be a second sched-pipeline consumer on the warp). Instead it returns the
        per-tile aux GMEM->SMEM producer body (``list[ast.stmt]``) for the
        CALLER to inject into the (widened) epilogue role-local while on the
        STORE warp, which already owns the single per-warp sched-pipeline
        handshake. This is how the store/epi-load warp does BOTH the early
        residual load and the late TMA-D store drain in one loop at 8 warps. No
        post-loop producer tail is returned (the aux producer_state advance is
        inside the shared loop's store-warp branch, so a post-loop tail would
        hit an IR domination error; the boundary drain is unnecessary because
        the consumers release all stages by loop exit). ``inline_aux_only`` uses
        the post-L2 ``tile_offset_0/1`` coords (always present in the epi loop's
        dependency stmts), so it asserts the post-L2 path.

        Active when the matmul plan has ``c_input_warp_count > 0``
        AND the forward FX walker discovered one or more
        auxiliary-tensor descriptors. The C-input warp participates
        in the scheduler-broadcast pipeline as a *consumer* of
        ``sched_pipeline`` (per-tile coord broadcast) and as a
        *producer* of the ``c_pipeline_aux`` SMEM aux ring
        (one cooperative ``cute.copy(GMEM_aux, SMEM_aux[stage])``
        per output tile per descriptor).

        Per-tile body:

        1. ``sched_pipeline.consumer_wait`` + valid-flag read
           (shared with ``_build_role_local_while_with_scheduler``
           via ``_build_sched_pipeline_consumer_{wait,release}_block``).
        2. ``virtual_pid_var`` write decomposed from
           ``work_tile_smem`` (downstream M / N tile coords used
           by the per-descriptor aux GMEM-tile builder).
        3. On the production post-L2 path,
           ``sched_pipeline.consumer_release`` runs after those
           coordinates are materialized, before aux staging, so the
           scheduler warp can run ahead. The defensive no-post-L2
           fallback keeps the release at the bottom because that path
           still reads the scheduler SMEM mailbox in the aux setup.
        4. Per output tile, build the per-CTA aux GMEM region:
           ``cute.local_tile(host_aux, (bm_per_cta, bn),
           (tile_m, tile_n))`` where
           ``bm_per_cta = bm // cluster_m`` under
           ``use_2cta_instrs`` (otherwise ``bm``). For rank-1
           trailing-axis broadcast aux the M extent is also
           ``bm_per_cta``. ``flat_divide(epi_tile)`` +
           ``group_modes(2, rank)`` to expose a flat subtile
           axis whose extent matches the consumer's per-CTA
           subtile count.
        5. Build the cooperative ``TiledCopy`` once per
           descriptor: ``make_tiled_copy_tv`` with a
           ``(M_threads=4, N_threads=8)`` ordered layout × a
           ``(1, 128 / dtype_bits)`` val layout and a
           ``CopyUniversalOp`` atom. ``get_slice(lane_idx)``
           per lane.
        6. Per subtile (``cutlass.range(subtile_count,
           unroll_full=True)``): ``producer_acquire(state)`` →
           per descriptor build the per-subtile GMEM slice and
           SMEM stage slice and issue
           ``cute.copy(tiled_copy, gmem_part, smem_part)`` →
           ``cute.arch.sync_warp()`` → ``cute.arch.fence_acq_rel_cta()``
           (so the consumer's generic SMEM reads after
           ``consumer_wait`` see the producer's stores —
           ``mbarrier.arrive`` from ``AsyncThread`` has relaxed
           memory semantics and does not fence by itself) →
           ``c_pipeline_aux.producer_commit(state)`` →
           ``state.advance()``.
        7. The sentinel-publish wait remains at the bottom. On the
           defensive no-post-L2 fallback, the delayed
           ``sched_pipeline.consumer_release`` runs just before this
           wait.

        The producer-side and consumer-side flip
        (in ``memory_ops._aux_subtile_load_source`` /
        ``_aux_tile_setup_lines``) MUST land in the same commit:
        a partial-handshake state deadlocks once a CTA wraps the
        pipeline depth (an early-2a variant of this builder
        emitted producer barriers without consumer releases; with
        ``num_stages=2`` the third ``producer_acquire`` blocks
        forever — see the cycle-2a docstring in
        ``TestCuteTcgen05AuxPipelineCycle2a``).
        """
        plan = self._tcgen05_plan()
        # The aux producer runs on the C-input warp normally; under the cycle-94
        # merge (``inline_aux_only``) it runs on the store warp instead, so the
        # producer-warp invariant is "C-input OR store warp present".
        assert plan is not None and (plan.has_c_input_warp or plan.has_store_warp), (
            "aux producer body requires a matmul plan with a C-input or store warp"
        )
        c_input_aux_tensor_descriptors = plan.c_input_aux_tensor_descriptors
        assert c_input_aux_tensor_descriptors, (
            "C-input role-local while requires non-empty exact-shape aux "
            "descriptors (producer-body split gate must be open)"
        )
        sched_pipeline_plan = self._tcgen05_sched_pipeline_plan()
        assert sched_pipeline_plan is not None, (
            "C-input role-local while requires a registered "
            "sched_pipeline plan; was cute_state.register_tcgen05_sched_pipeline_plan "
            "called by _codegen_cute_mma?"
        )
        sched_pipeline = sched_pipeline_plan.pipeline
        sched_consumer_state = sched_pipeline_plan.consumer_state
        # Aux pipeline plan: the matmul-plan gate that admits this
        # builder also fires the pipeline allocation in
        # ``cute_mma._codegen_cute_mma``, so a non-None plan is
        # the invariant we rely on once the gate is open. The
        # assert catches a future gate-skew between
        # ``has_c_input_warp + c_input_aux_tensor_descriptors`` and the
        # cute_mma allocator rather than producing a half-allocated
        # kernel.
        aux_pipeline_plan = device_function.cute_state.aux_pipeline_plan
        assert aux_pipeline_plan is not None, (
            "C-input role-local while requires a registered "
            "aux pipeline plan; was cute_state.register_tcgen05_aux_pipeline_plan "
            "called by _codegen_cute_mma?"
        )
        assert len(aux_pipeline_plan.rings) == len(c_input_aux_tensor_descriptors), (
            "C-input role-local while: aux pipeline plan must have one "
            "ring per staged matmul-plan aux descriptor"
        )
        aux_use_tma_load = aux_pipeline_plan.use_tma_load
        aux_requires_full_tile = plan.tma_store_full_tiles_only

        valid_var = device_function.new_var("tcgen05_c_input_warp_valid")
        work_tile_stage_index = (
            f"{sched_consumer_state}.index"
            if self._tcgen05_uses_staged_work_tile_mailbox()
            else None
        )

        if aux_requires_full_tile and tile_phase == "edge":
            # Edge epilogues use the SIMT direct-GMEM aux path. The C-input
            # warp still participates in the scheduler pipeline so the
            # producer's consumer-arrival count remains balanced, but each
            # edge iteration is purely the sched-pipeline handshake: wait for
            # the broadcast, release it, then wait for the next one.
            prelude: list[ast.stmt] = []
            prelude.extend(
                _build_sched_pipeline_consumer_wait_block(
                    sched_pipeline=sched_pipeline,
                    sched_consumer_state=sched_consumer_state,
                    work_tile_smem=layout.work_tile_smem,
                    valid_var=valid_var,
                    work_tile_stage_index=work_tile_stage_index,
                )
            )
            per_tile_body: list[ast.stmt] = []
            per_tile_body.extend(
                _build_sched_pipeline_consumer_release_block(
                    sched_pipeline=sched_pipeline,
                    sched_consumer_state=sched_consumer_state,
                )
            )
            per_tile_body.extend(
                _build_sched_pipeline_consumer_wait_block(
                    sched_pipeline=sched_pipeline,
                    sched_consumer_state=sched_consumer_state,
                    work_tile_smem=layout.work_tile_smem,
                    valid_var=valid_var,
                    work_tile_stage_index=work_tile_stage_index,
                )
            )
            prelude.append(
                create(
                    ast.While,
                    test=expr_from_string(valid_var),
                    body=per_tile_body,
                    orelse=[],
                )
            )
            prelude.extend(
                _build_sched_pipeline_consumer_release_block(
                    sched_pipeline=sched_pipeline,
                    sched_consumer_state=sched_consumer_state,
                )
            )
            return create(
                ast.If,
                test=expr_from_string(self._tcgen05_c_input_role_predicate()),
                body=prelude,
                orelse=[],
            )

        coord_terms = [
            self._tcgen05_work_tile_slot(layout, i) for i in range(len(self.pid_info))
        ]
        linear_pid_expr = self._tcgen05_linear_virtual_pid_from_coords_expr(coord_terms)
        sched_coord_0 = coord_terms[0] if len(coord_terms) > 0 else "cutlass.Int32(0)"
        sched_coord_1 = coord_terms[1] if len(coord_terms) > 1 else "cutlass.Int32(0)"

        # M / N tile coords for the cooperative copy. Each CTA's
        # C-input warp loads only its own per-CTA portion of the
        # aux region. Under cluster_m=2 ``use_2cta_instrs`` the
        # per-CTA aux tile shape is ``(bm/2, bn)`` and the per-CTA
        # M tile coord is the global M tile (without the cluster
        # ``// 2`` reduction — each CTA in the cluster handles its
        # own row stripe). The N coord is shared across cluster
        # (cluster_n=1 is the only validated runtime).
        #
        # Critical correctness invariant: the consumer-side per-CTA
        # subtile count is
        # ``(bm_per_cta * bn) / (epi_m * epi_n)``; the producer's
        # subtile count must match or the mbar handshake deadlocks
        # once the producer wraps the stage count and the consumer
        # has already exited (cluster_m=2 yields producer=2× the
        # consumer count when the producer mistakenly uses the
        # cluster-level (bm, bn)).
        bm = plan.bm
        bn = plan.bn
        cluster_m = self._tcgen05_cluster_m()
        is_two_cta = self._tcgen05_is_two_cta()
        bm_per_cta = bm // cluster_m if is_two_cta else bm
        tile_m_var = device_function.new_var("tcgen05_aux_tile_m")
        tile_n_var = device_function.new_var("tcgen05_aux_tile_n")
        # Bring the L2-grouping PID-decomposition chain (the
        # ``inner_2d_pid`` → ``pid_0`` / ``pid_1`` →
        # ``tile_offset_0`` / ``tile_offset_1`` line) into this
        # role-local while body so the producer's per-CTA aux
        # GMEM tile aligns with the consumer's post-L2-remap
        # logical tile coords. Without it the producer would
        # build its per-CTA aux GMEM tile from the raw
        # ``work_tile_smem`` coords (which equal the consumer's
        # only under ``l2_groupings=[1]``); under
        # ``l2_groupings=[g>1]`` the consumer's post-L2-remap
        # ``pid_0`` / ``pid_1`` no longer equal ``work_tile_smem[0,1]``
        # and the producer fetches a misaligned aux tile.
        # ``_role_local_dependency_stmts`` walks the shared body
        # backward from a synthetic read of ``tile_offset_0`` /
        # ``tile_offset_1`` and returns the smallest set of
        # statements that define them. The walker is the same
        # one the consumer role-local whiles use; this keeps the
        # producer and consumer in lockstep on whatever the
        # L2-grouping decomposition emits.
        #
        # ``tile_offset_0`` / ``tile_offset_1`` are emitted
        # unconditionally by the standard ``NDTileStrategy``
        # decomposition (see ``tile_strategy.py:_strategy_codegen``
        # — ``tile_offset_<i> = pid_<i> * BS`` is part of every
        # tile body). They are therefore always present in
        # ``shared_body_extracted`` for any real kernel binding,
        # regardless of whether ``L2GroupingProgramIDs.codegen``
        # wraps the strategy (l2_grp=[g>1]) or not (l2_grp=[1]
        # passes the names through directly with the identity
        # remap). The branch on ``has_post_l2_coords`` below is
        # purely defensive — it preserves the pre-cycle-2i
        # ``work_tile_smem`` fallback for the hypothetical case
        # where a future strategy emits the role-local while
        # without these names, so the cycle 2b correctness
        # baseline at ``l2_grp=[1]`` cannot regress silently.
        synthetic_reads_for_l2 = [
            statement_from_string(
                "_tcgen05_aux_l2_anchor = tile_offset_0 + tile_offset_1"
            )
        ]
        l2_dependency_stmts: list[ast.stmt] = []
        if shared_body_extracted is not None:
            l2_dependency_stmts = self._role_local_dependency_stmts(
                shared_body_extracted, synthetic_reads_for_l2
            )
        l2_dependency_writes: set[str] = set()
        for stmt in l2_dependency_stmts:
            _, writes = _stmt_name_uses(stmt)
            l2_dependency_writes.update(writes)
        has_post_l2_coords = (
            "tile_offset_0" in l2_dependency_writes
            and "tile_offset_1" in l2_dependency_writes
        )
        # ``peer_m`` is this CTA's rank along the M axis of the cluster:
        # ``block_idx_in_cluster() % cluster_m``. The modulo is load-bearing
        # here because consumer CTAs can span the N axis: for ``cluster_m=2 +
        # cluster_n=2 + use_2cta=True`` (a validated 4-CTA cluster shape, see
        # ``cute_mma.py:_TCGEN05_V_LEADER_PREDICATE``) ranks {0, 1, 2, 3}
        # have M peer ranks {0, 1, 0, 1}. Scheduler-warp broadcasts use
        # lane-rank branches restricted to ``peer_rank < cluster_m``, where
        # the raw lane rank is already the M peer rank.
        peer_m_expr = (
            f"(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
            f"% cutlass.Int32({cluster_m}))"
        )
        if has_post_l2_coords:
            # Post-L2 path. ``tile_offset_0 // bm`` is the
            # post-L2-remap logical M tile index (== ``pid_0``
            # in the decomposition emitted right above this
            # body); ``tile_offset_1 // bn`` is ``pid_1``.
            #
            # Note that under ``cluster_n=1 + l2_groupings=[1]``
            # the post-L2 expression ``pid_0 * cluster_m +
            # peer_m`` is numerically equal to the pre-cycle-2i
            # ``work_tile_smem[0]`` because (a) the L2 remap is
            # identity and (b) the scheduler publishes
            # ``tile_idx[0] + peer_m = pid_0 * cluster_m +
            # peer_m`` into each CTA's slot. Outside that
            # narrow case the two forms diverge: under
            # ``l2_grp=[g>1]`` because L2 remap is non-identity,
            # and under ``cluster_n=2`` because the raw
            # rank-in-cluster ≠ peer_m.
            m_source = f"(tile_offset_0 // cutlass.Int32({bm}))"
            n_source = f"(tile_offset_1 // cutlass.Int32({bn}))"
            if is_two_cta:
                tile_m_expr = (
                    f"({m_source}) * cutlass.Int32({cluster_m}) + {peer_m_expr}"
                )
            elif self._tcgen05_uses_cluster_m2_one_cta_role_local_bridge():
                # Bridge: cluster has ``cluster_m`` CTAs each
                # handling its own logical M tile (no V-pair
                # striping); add peer_m to step from the
                # cluster-leader CTA's logical tile to this
                # CTA's. Bridge requires ``cluster_n=1`` (see
                # ``cute_mma._tcgen05_cluster_m2_one_cta_role_local_bridge``)
                # so ``peer_m == block_idx_in_cluster()``; using
                # ``peer_m`` form keeps the expression
                # robust if that constraint widens later.
                tile_m_expr = f"({m_source}) + {peer_m_expr}"
            else:
                tile_m_expr = m_source
            tile_n_expr = n_source
        else:
            # Pre-cycle-2i raw scheduler coords. Unreachable in
            # production today: ``tile_offset_0`` /
            # ``tile_offset_1`` are emitted unconditionally by
            # the standard ``NDTileStrategy`` decomposition,
            # which runs for every real kernel binding
            # regardless of ``l2_groupings``. Purely defensive —
            # preserves the pre-cycle-2i correctness baseline
            # if a future strategy emits the role-local while
            # without those names. Under ``is_two_cta`` the
            # scheduler-published ``work_tile_smem[0]`` already
            # carries the per-CTA peer_m baked in (the
            # scheduler publish is
            # ``tile_idx[0] + peer_m`` per CTA); under non-2cta
            # ``_tcgen05_logical_m_coord_expr`` adds the bridge
            # adjustment when applicable. So no extra peer_m
            # is applied here.
            m_source = sched_coord_0
            n_source = sched_coord_1
            if is_two_cta:
                tile_m_expr = m_source
            else:
                tile_m_expr = self._tcgen05_logical_m_coord_expr(m_source)
            tile_n_expr = n_source

        # Cooperative-copy thread layout. Single C-input warp has
        # 32 lanes; lay them out row-major as (M_threads, N_threads)
        # = (4, 8) so each lane reads a contiguous N chunk
        # (innermost dim) and 8 lanes cover the N axis. The val
        # layout pulls 128 bits per copy atom — the largest power
        # of two that divides both ``bn * dtype_bits`` and 128. For
        # bf16 with bn=256 that's 128 bits = 8 elements.
        # ``make_tiled_copy_tv`` lifts the per-lane chunk to a
        # ``(thr_layout × val_layout)`` partition and ``cute.copy``
        # iterates the tile under the lane's get_slice(lane_idx).
        cute_lane_idx_var = device_function.new_var("tcgen05_aux_lane_idx")

        # Factories mirror the consumer-side pattern in
        # ``_build_role_local_while_with_scheduler`` — both go
        # through the shared module-scope helpers
        # (``_build_sched_pipeline_consumer_{wait,release}_block``)
        # so the wait/release shape has one source of truth.
        def _sched_consumer_wait_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_wait_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
                work_tile_smem=layout.work_tile_smem,
                valid_var=valid_var,
                work_tile_stage_index=work_tile_stage_index,
            )

        def _sched_consumer_release_block() -> list[ast.stmt]:
            return _build_sched_pipeline_consumer_release_block(
                sched_pipeline=sched_pipeline,
                sched_consumer_state=sched_consumer_state,
            )

        # Pull aux pipeline names from the plan. The plan is the
        # ``_Tcgen05AuxPipelinePlan`` dataclass; access by name
        # without importing the type to avoid a module cycle.
        aux_pipeline_name = aux_pipeline_plan.pipeline
        aux_producer_state_name = aux_pipeline_plan.producer_state
        aux_rings = aux_pipeline_plan.rings
        aux_epi_tile_var = aux_pipeline_plan.epi_tile_var
        aux_tma_barrier_var = (
            device_function.new_var("tcgen05_aux_tma_barrier")
            if aux_use_tma_load
            else None
        )

        aux_full_tile_var = device_function.new_var("tcgen05_aux_full_tile")
        aux_shape = c_input_aux_tensor_descriptors[0].host_tensor_val.shape
        assert len(aux_shape) == 2, (
            "C-input staged aux descriptors must be exact-shape rank-2 tensors"
        )
        aux_m_size = int(aux_shape[0])
        aux_n_size = int(aux_shape[1])
        if has_post_l2_coords:
            aux_m_start_expr = "tile_offset_0"
            aux_n_start_expr = "tile_offset_1"
        else:
            if is_two_cta:
                aux_m_tile_expr = f"({sched_coord_0} // cutlass.Int32({cluster_m}))"
            else:
                aux_m_tile_expr = sched_coord_0
            aux_m_start_expr = f"({aux_m_tile_expr}) * cutlass.Int32({bm})"
            aux_n_start_expr = f"{sched_coord_1} * cutlass.Int32({bn})"
        aux_full_tile_expr = (
            f"{aux_m_start_expr} + cutlass.Int32({bm}) "
            f"<= cutlass.Int32({aux_m_size}) and "
            f"{aux_n_start_expr} + cutlass.Int32({bn}) "
            f"<= cutlass.Int32({aux_n_size})"
        )

        env_backend = CompileEnvironment.current().backend

        # Per-descriptor partitioning that runs once per output tile:
        # builds the source 2-D GMEM tensor, slices the per-output-
        # tile ``(bm, bn)`` region, and flat-divides it into
        # epi-tile-sized subtiles. The subtile-loop body further
        # slices one subtile of GMEM and one stage of SMEM per
        # iteration. Each per-descriptor partition uses fresh AST
        # var names so multiple aux descriptors compose linearly.
        per_descriptor_setup_blocks: list[list[ast.stmt]] = []
        per_descriptor_subtile_blocks: list[list[str]] = []
        per_descriptor_grouped_names: list[str] = []
        # Same N-threads = 8 / M-threads = 4 layout the producer uses
        # for the cooperative copy. For the matmul-plan epi tile (a
        # rectangular sub-tile of the (bm, bn) region) the lane
        # layout is constexpr and shared across descriptors.
        n_threads = 8
        m_threads = 32 // n_threads
        for desc_idx, (desc, ring) in enumerate(
            zip(c_input_aux_tensor_descriptors, aux_rings, strict=True)  # type: ignore[arg-type]
        ):
            aux_tensor_name = device_function.tensor_arg(desc.host_tensor_val).name
            aux_dtype_str = env_backend.dtype_str(desc.host_tensor_val.dtype)
            dtype_bits = desc.host_tensor_val.dtype.itemsize * 8
            copy_bits = 128
            num_copy_elems = max(1, copy_bits // dtype_bits)
            tma_atom = ring.tma_atom
            tma_tensor = ring.tma_tensor
            if aux_use_tma_load:
                assert tma_atom is not None
                assert tma_tensor is not None
                assert aux_tma_barrier_var is not None
            else:
                assert tma_atom is None
                assert tma_tensor is None
            gmem_aux_view_var = device_function.new_var(
                f"tcgen05_aux_gmem_view_{desc_idx}"
            )
            gmem_aux_tile_var = device_function.new_var(
                f"tcgen05_aux_gmem_tile_{desc_idx}"
            )
            gmem_aux_subtiles_var = device_function.new_var(
                f"tcgen05_aux_gmem_subtiles_{desc_idx}"
            )
            gmem_subtiles_grouped_var = device_function.new_var(
                f"tcgen05_aux_gmem_subtiles_grouped_{desc_idx}"
            )
            tiled_copy_var = device_function.new_var(
                f"tcgen05_aux_tiled_copy_{desc_idx}"
            )
            thr_copy_var = device_function.new_var(f"tcgen05_aux_thr_copy_{desc_idx}")
            gmem_subtile_var = device_function.new_var(
                f"tcgen05_aux_gmem_subtile_{desc_idx}"
            )
            smem_stage_var = device_function.new_var(
                f"tcgen05_aux_smem_stage_{desc_idx}"
            )
            gmem_part_var = device_function.new_var(f"tcgen05_aux_gmem_part_{desc_idx}")
            smem_part_var = device_function.new_var(f"tcgen05_aux_smem_part_{desc_idx}")
            tma_smem_part_var = ""
            tma_gmem_part_var = ""
            setup: list[ast.stmt] = []
            # Build the source 2-D GMEM tensor. Exact-shape rank-2
            # aux passes through ``aux_tensor`` directly; rank-1
            # trailing-axis broadcast aux builds a stride-0-on-M
            # view with M-extent ``bm_per_cta`` and N-extent = the
            # rank-1 size. Under cluster_m=2 ``use_2cta_instrs``
            # the per-CTA aux tile is ``(bm/2, bn)``; the global M
            # tile coord directly indexes per-CTA M stripes (the
            # scheduler publishes 2 global tiles per cluster
            # step). For non-2cta the per-CTA tile shape collapses
            # to the full ``(bm, bn)``.
            if desc.broadcast_axis is None:
                if aux_use_tma_load:
                    gmem_source_var = tma_tensor
                    assert gmem_source_var is not None
                else:
                    gmem_source_var = aux_tensor_name
                setup.extend(
                    [
                        statement_from_string(
                            f"{gmem_aux_view_var} = {gmem_source_var}"
                        ),
                        statement_from_string(
                            f"{gmem_aux_tile_var} = cute.local_tile("
                            f"{gmem_aux_view_var}, ({bm_per_cta}, {bn}), "
                            f"({tile_m_var}, {tile_n_var}))"
                        ),
                    ]
                )
            else:
                assert desc.broadcast_axis == 1, (
                    "C-input warp aux producer expects "
                    "broadcast_axis in {None, 1}; the chain "
                    "analyzer rejects other forms"
                )
                n_global = int(desc.host_tensor_val.shape[0])
                setup.extend(
                    [
                        statement_from_string(
                            f"{gmem_aux_view_var} = cute.make_tensor("
                            f"{aux_tensor_name}.iterator, "
                            f"cute.make_layout(({bm_per_cta}, {n_global}), "
                            "stride=(0, 1)))"
                        ),
                        statement_from_string(
                            f"{gmem_aux_tile_var} = cute.local_tile("
                            f"{gmem_aux_view_var}, ({bm_per_cta}, {bn}), "
                            f"(cutlass.Int32(0), {tile_n_var}))"
                        ),
                    ]
                )
            # Subdivide the per-output-tile aux region into epi-tile-
            # sized subtiles using ``flat_divide(epi_tile)`` —
            # mirrors the consumer-side ``flat_divide`` so the
            # producer and consumer iterate the same subtile
            # ordering. ``group_modes(..., 2, rank)`` collapses the
            # outer (subtile_m, subtile_n) modes into one linear
            # subtile axis so the producer's subtile loop sees a
            # flat ``subtile_count`` extent (matches the consumer's
            # post-``group_modes`` shape used inside
            # ``_aux_subtile_load_source``).
            setup.extend(
                [
                    statement_from_string(
                        f"{gmem_aux_subtiles_var} = cute.flat_divide("
                        f"{gmem_aux_tile_var}, {aux_epi_tile_var})"
                    ),
                    statement_from_string(
                        f"{gmem_subtiles_grouped_var} = cute.group_modes("
                        f"{gmem_aux_subtiles_var}, 2, "
                        f"cute.rank({gmem_aux_subtiles_var}))"
                    ),
                ]
            )
            if aux_use_tma_load:
                tma_smem_part_var = device_function.new_var(
                    f"tcgen05_aux_tma_smem_part_{desc_idx}"
                )
                tma_gmem_part_var = device_function.new_var(
                    f"tcgen05_aux_tma_gmem_part_{desc_idx}"
                )
                setup.append(
                    statement_from_string(
                        f"{tma_smem_part_var}, {tma_gmem_part_var} = "
                        "cute.nvgpu.cpasync.tma_partition("
                        f"{tma_atom}, 0, cute.make_layout(1), "
                        f"cute.group_modes({ring.smem}, 0, "
                        f"cute.rank({ring.smem}) - 1), "
                        f"cute.group_modes({gmem_subtiles_grouped_var}, 0, "
                        f"cute.rank({gmem_subtiles_grouped_var}) - 1))"
                    )
                )
            else:
                # Cooperative-copy ``TiledCopy`` for the C-input warp's
                # 32 lanes. The atom uses ``CopyUniversalOp`` (regular
                # SIMT ld+st): cp.async would impose a 128-bit
                # source-iterator alignment check that the layout-
                # implied stride alignment cannot satisfy at IR-build
                # time (the host pointer is 16-byte aligned, but the
                # minimum row stride is 1 element = 16 bits for bf16).
                # ``CopyUniversalOp`` lowers to a SIMT ld/st pair whose
                # vectorization is driven at runtime by the host
                # pointer's actual alignment.
                setup.extend(
                    [
                        statement_from_string(
                            f"{tiled_copy_var} = cute.make_tiled_copy_tv("
                            f"cute.make_copy_atom("
                            f"cute.nvgpu.CopyUniversalOp(), {aux_dtype_str}, "
                            f"num_bits_per_copy={copy_bits}), "
                            f"cute.make_ordered_layout("
                            f"({m_threads}, {n_threads}), order=(1, 0)), "
                            f"cute.make_layout((1, {num_copy_elems})))"
                        ),
                        statement_from_string(
                            f"{thr_copy_var} = {tiled_copy_var}.get_slice("
                            f"{cute_lane_idx_var})"
                        ),
                    ]
                )
            per_descriptor_setup_blocks.append(setup)
            per_descriptor_grouped_names.append(gmem_subtiles_grouped_var)

            # Per-subtile body source: builds the per-stage GMEM
            # slice + SMEM stage slice and issues one cooperative
            # ``cute.copy``. The loop variable
            # ``_tcgen05_aux_subtile`` indexes the flat subtile
            # axis (collapsed via ``group_modes(..., 2, rank)``);
            # the consumer's per-subtile loop indexes the same
            # axis identically.
            if aux_use_tma_load:
                subtile_lines = [
                    (
                        f"cute.copy({tma_atom}, "
                        f"{tma_gmem_part_var}[None, "
                        f"cutlass.Int32(_tcgen05_aux_subtile)], "
                        f"{tma_smem_part_var}[None, "
                        f"{aux_producer_state_name}.index], "
                        f"tma_bar_ptr={aux_tma_barrier_var})"
                    ),
                ]
            else:
                subtile_lines = [
                    (
                        f"{gmem_subtile_var} = "
                        f"{gmem_subtiles_grouped_var}[None, None, "
                        f"cutlass.Int32(_tcgen05_aux_subtile)]"
                    ),
                    (
                        f"{smem_stage_var} = {ring.smem}[None, None, "
                        f"{aux_producer_state_name}.index]"
                    ),
                    (
                        f"{gmem_part_var} = "
                        f"{thr_copy_var}.partition_S({gmem_subtile_var})"
                    ),
                    (f"{smem_part_var} = {thr_copy_var}.partition_D({smem_stage_var})"),
                    (f"cute.copy({tiled_copy_var}, {gmem_part_var}, {smem_part_var})"),
                ]
            per_descriptor_subtile_blocks.append(subtile_lines)

        def _aux_copy_lines() -> list[ast.stmt]:
            """Emit the per-output-tile producer body.

            The body computes per-output-tile aux GMEM partitions,
            flat-divides each into epi-tile-sized subtiles, then
            loops over the subtile axis. Each subtile iteration
            acquires one SMEM ring stage, cooperative-copies the
            subtile of every descriptor into the stage, fences the
            SMEM proxy, commits the producer barrier, and advances
            the producer state. Iteration order matches the
            consumer's per-subtile loop in
            ``memory_ops._aux_subtile_load_source``.
            """
            lines: list[ast.stmt] = []
            lines.extend(
                [
                    statement_from_string(
                        f"{cute_lane_idx_var} = cute.arch.lane_idx()"
                    ),
                    statement_from_string(f"{tile_m_var} = {tile_m_expr}"),
                    statement_from_string(f"{tile_n_var} = {tile_n_expr}"),
                ]
            )
            # Per-descriptor partition setup runs once per output
            # tile, before the subtile loop. The setup builds the
            # ``flat_divide(epi_tile) → group_modes(2, rank)``
            # tensor whose third mode is the subtile axis the
            # producer iterates against.
            for block in per_descriptor_setup_blocks:
                lines.extend(block)

            # Determine the subtile count from any descriptor's
            # grouped tensor (all descriptors share the same
            # subtile axis because they're all sliced from the
            # same ``(bm, bn)`` region with the same ``epi_tile``).
            # Use the first descriptor's grouped name — pulled
            # from ``per_descriptor_grouped_names`` so the
            # ``device_function.new_var`` namespace suffix
            # (if any) is honored.
            first_grouped = per_descriptor_grouped_names[0]
            subtile_count_var = device_function.new_var(
                "tcgen05_aux_producer_subtile_count"
            )
            lines.append(
                statement_from_string(
                    f"{subtile_count_var} = cutlass.const_expr("
                    f"cute.size({first_grouped}.shape, mode=[2]))"
                )
            )

            # Build the per-subtile loop body. Per iteration:
            # acquire one SMEM stage, copy every descriptor's
            # subtile into the stage, sync the warp + fence the
            # SMEM proxy so the consumer's
            # ``consumer_wait`` sees a fully populated stage,
            # commit, advance. Lane-uniform code throughout (every
            # lane runs the same per-iteration body; the cooperative
            # copy partitions inside).
            # Build the per-iteration body as a single
            # already-indented source string. Each entry in
            # ``inner_chunks`` is a top-level statement (or block)
            # carrying its own ``    `` indent for the surrounding
            # ``for ...:`` loop. ``emit_pipeline_advance`` may
            # return a multi-line ``if True: ...`` block on
            # cutedsl builds without the OpResultList fix; we
            # pass ``indent="    "`` so the whole block is
            # already indented for the loop body and no caller-
            # side reflow is needed (the prior single-line
            # ``\n.join(f"    {line}" ...)`` pattern only
            # indented the first line of the advance block,
            # under-indenting its body and causing a SyntaxError
            # on the fallback path).
            loop_indent = "    "
            inner_chunks: list[str] = []
            inner_chunks.append(
                f"{loop_indent}{aux_pipeline_name}.producer_acquire("
                f"{aux_producer_state_name})"
            )
            if aux_use_tma_load:
                assert aux_tma_barrier_var is not None
                inner_chunks.append(
                    f"{loop_indent}{aux_tma_barrier_var} = "
                    f"{aux_pipeline_name}.producer_get_barrier("
                    f"{aux_producer_state_name})"
                )
            for block in per_descriptor_subtile_blocks:
                inner_chunks.extend(f"{loop_indent}{line}" for line in block)
            # TMA aux loads skip the SIMT warp sync/fence below: they are
            # ordered by the tx-counted PipelineTmaAsync barrier, and the
            # consumer fences the async-shared view after ``consumer_wait`` and
            # before generic SMEM reads.
            if not aux_use_tma_load:
                # ``CopyUniversalOp`` issues regular ld+st pairs that
                # complete in program order per thread; ``sync_warp``
                # ensures all 32 lanes of the producer warp finish
                # their SMEM stores. ``fence_acq_rel_cta`` provides
                # cross-warp visibility through the CTA-scope generic
                # SMEM proxy — the consumer warps' generic SMEM reads
                # after their ``consumer_wait`` would otherwise be
                # free to bypass the producer's writes since the
                # AsyncThread ``mbarrier.arrive`` PTX emission has
                # relaxed memory semantics by default.
                inner_chunks.extend(
                    [
                        f"{loop_indent}cute.arch.sync_warp()",
                        f"{loop_indent}cute.arch.fence_acq_rel_cta()",
                    ]
                )
            inner_chunks.extend(
                [
                    (
                        f"{loop_indent}{aux_pipeline_name}.producer_commit("
                        f"{aux_producer_state_name})"
                    ),
                    emit_pipeline_advance(aux_producer_state_name, indent=loop_indent),
                ]
            )
            inner_body = "\n".join(inner_chunks)
            loop_src = (
                f"for _tcgen05_aux_subtile in cutlass.range("
                f"{subtile_count_var}, unroll_full=True):\n"
                f"{inner_body}"
            )
            lines.append(statement_from_string(loop_src))
            return lines

        if inline_aux_only:
            # Cycle-94 merge: return the per-tile aux producer body for injection
            # into the store warp's branch of the (widened) epilogue role-local
            # while. The epi loop owns the sched handshake and materializes the
            # post-L2 tile coords, so no sched wait/release is emitted here. The
            # full-tile guard mirrors the standalone builder so edge tiles (SIMT
            # aux) commit no aux stages the consumer never releases.
            #
            # No post-loop ``producer_tail`` is returned: the aux producer_state
            # is advanced inside the store-warp branch of the SHARED epilogue
            # while, so a post-loop ``producer_tail(state)`` would reference a
            # loop-carried value defined in that nested region (IR domination
            # error). The tail is only a boundary drain for the TMA-load empty
            # barriers, and the epi-warp consumers have already released every
            # aux stage by loop exit, so it is safely omitted on the merge path.
            assert has_post_l2_coords, (
                "inline_aux_only merge requires post-L2 tile coords "
                "(tile_offset_0/1) from the epilogue loop's dependency stmts"
            )
            inline_per_tile: list[ast.stmt] = []
            aux_copy_lines_inline = _aux_copy_lines()
            if aux_requires_full_tile:
                inline_per_tile.extend(
                    [
                        statement_from_string(
                            f"{aux_full_tile_var} = {aux_full_tile_expr}"
                        ),
                        create(
                            ast.If,
                            test=expr_from_string(aux_full_tile_var),
                            body=aux_copy_lines_inline,
                            orelse=[],
                        ),
                    ]
                )
            else:
                inline_per_tile.extend(aux_copy_lines_inline)
            return inline_per_tile

        prelude: list[ast.stmt] = []
        prelude.extend(_sched_consumer_wait_block())
        per_tile_body: list[ast.stmt] = [
            statement_from_string(f"{self.virtual_pid_var} = {linear_pid_expr}"),
        ]
        # Emit the L2-grouping decomposition chain right after
        # ``virtual_pid`` is bound, so ``_aux_copy_lines()`` can
        # reference the post-L2 ``tile_offset_0`` / ``tile_offset_1``
        # names. Mirrors the consumer role-local while body's
        # placement of ``dependency_stmts`` (see
        # ``_build_role_local_while_with_scheduler``).
        per_tile_body.extend(l2_dependency_stmts)
        early_sched_release = has_post_l2_coords
        if early_sched_release:
            # After the post-L2 coordinate chain has materialized tile_offset_*
            # locals, the aux producer no longer needs the scheduler SMEM
            # mailbox for this tile. Release here so the scheduler warp can run
            # ahead while aux GMEM->SMEM staging is in flight.
            per_tile_body.extend(_sched_consumer_release_block())
        aux_copy_lines = _aux_copy_lines()
        if aux_requires_full_tile and tile_phase == "all":
            # Hybrid full-tile TMA store with a SIMT edge fallback uses the aux
            # SMEM ring only on full tiles. Edge tiles take direct-GMEM aux
            # loads, so the producer must skip them; otherwise it commits aux
            # stages that no consumer will release.
            per_tile_body.extend(
                [
                    statement_from_string(
                        f"{aux_full_tile_var} = {aux_full_tile_expr}"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(aux_full_tile_var),
                        body=aux_copy_lines,
                        orelse=[],
                    ),
                ]
            )
        else:
            assert tile_phase in ("all", "full"), f"unexpected tile_phase={tile_phase}"
            per_tile_body.extend(aux_copy_lines)
        if not early_sched_release:
            per_tile_body.extend(_sched_consumer_release_block())
        per_tile_body.extend(_sched_consumer_wait_block())
        prelude.append(
            create(
                ast.While,
                test=expr_from_string(valid_var),
                body=per_tile_body,
                orelse=[],
            )
        )
        prelude.extend(_sched_consumer_release_block())
        if aux_use_tma_load:
            prelude.append(
                statement_from_string(
                    f"{aux_pipeline_name}.producer_tail({aux_producer_state_name})"
                )
            )

        return create(
            ast.If,
            test=expr_from_string(self._tcgen05_c_input_role_predicate()),
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

    @staticmethod
    def _tcgen05_is_local_assignment_target(target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return isinstance(target.ctx, ast.Store)
        if isinstance(target, ast.Tuple | ast.List):
            return all(
                Tcgen05PersistentProgramIDs._tcgen05_is_local_assignment_target(elt)
                for elt in target.elts
            )
        return False

    _TCGEN05_OMIT_SHARED_PURE_CALLS: ClassVar[frozenset[str]] = frozenset(
        {
            "range",
            "max",
            "min",
            "cutlass.BFloat16",
            "cutlass.Boolean",
            "cutlass.Float16",
            "cutlass.Float32",
            "cutlass.Float8E4M3FN",
            "cutlass.Int32",
            "cutlass.Int64",
            "cutlass.Uint8",
            # Scalar fallback loads left behind after all work-producing
            # statements move into role-local tcgen05 loops are side-effect
            # free. They may be discarded when none of their results feed
            # post-loop cleanup; stores and pipeline operations remain unsafe.
            "cute.arch.load",
        }
    )

    @staticmethod
    def _tcgen05_call_path(func: ast.AST) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = Tcgen05PersistentProgramIDs._tcgen05_call_path(func.value)
            if base is None:
                return None
            return f"{base}.{func.attr}"
        return None

    @staticmethod
    def _tcgen05_is_iterator_load(expr: ast.Call) -> bool:
        return (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "load"
            and not expr.args
            and not expr.keywords
            and any(
                isinstance(node, ast.Attribute) and node.attr == "iterator"
                for node in ast.walk(expr.func.value)
            )
        )

    @classmethod
    def _tcgen05_expr_safe_to_omit(cls, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Constant):
            return True
        if isinstance(expr, ast.Name):
            return isinstance(expr.ctx, ast.Load)
        if isinstance(expr, ast.Attribute):
            return cls._tcgen05_expr_safe_to_omit(expr.value)
        if isinstance(expr, ast.BinOp):
            return cls._tcgen05_expr_safe_to_omit(
                expr.left
            ) and cls._tcgen05_expr_safe_to_omit(expr.right)
        if isinstance(expr, ast.UnaryOp):
            return cls._tcgen05_expr_safe_to_omit(expr.operand)
        if isinstance(expr, ast.BoolOp):
            return all(cls._tcgen05_expr_safe_to_omit(value) for value in expr.values)
        if isinstance(expr, ast.Compare):
            return cls._tcgen05_expr_safe_to_omit(expr.left) and all(
                cls._tcgen05_expr_safe_to_omit(comparator)
                for comparator in expr.comparators
            )
        if isinstance(expr, ast.IfExp):
            return (
                cls._tcgen05_expr_safe_to_omit(expr.test)
                and cls._tcgen05_expr_safe_to_omit(expr.body)
                and cls._tcgen05_expr_safe_to_omit(expr.orelse)
            )
        if isinstance(expr, ast.Tuple | ast.List | ast.Set):
            return all(cls._tcgen05_expr_safe_to_omit(elt) for elt in expr.elts)
        if isinstance(expr, ast.Dict):
            return all(
                key is not None and cls._tcgen05_expr_safe_to_omit(key)
                for key in expr.keys
            ) and all(cls._tcgen05_expr_safe_to_omit(value) for value in expr.values)
        if isinstance(expr, ast.Subscript):
            return cls._tcgen05_expr_safe_to_omit(
                expr.value
            ) and cls._tcgen05_expr_safe_to_omit(expr.slice)
        if isinstance(expr, ast.Slice):
            return all(
                part is None or cls._tcgen05_expr_safe_to_omit(part)
                for part in (expr.lower, expr.upper, expr.step)
            )
        if isinstance(expr, ast.Call):
            # Generated scalar fallback loads can appear either as the
            # canonical ``cute.arch.load(...)`` helper or as a zero-argument
            # ``iterator.load()`` method call. Both are side-effect free; the
            # surrounding residual-body and post-loop dependency checks still
            # retain the shared loop whenever the loaded value is observed.
            if cls._tcgen05_is_iterator_load(expr):
                assert isinstance(expr.func, ast.Attribute)
                return cls._tcgen05_expr_safe_to_omit(expr.func.value)
            call_path = cls._tcgen05_call_path(expr.func)
            if call_path in {"max", "min"} and expr.keywords:
                return False
            return (
                call_path in cls._TCGEN05_OMIT_SHARED_PURE_CALLS
                and all(cls._tcgen05_expr_safe_to_omit(arg) for arg in expr.args)
                and all(
                    keyword.arg is not None
                    and cls._tcgen05_expr_safe_to_omit(keyword.value)
                    for keyword in expr.keywords
                )
            )
        return False

    @classmethod
    def _tcgen05_is_bare_sync_threads_call(cls, expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and cls._tcgen05_call_path(expr.func) == "cute.arch.sync_threads"
            and not expr.args
            and not expr.keywords
        )

    @staticmethod
    def _tcgen05_single_name_assignment(
        stmt: ast.stmt,
    ) -> tuple[str, ast.expr] | None:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.targets[0].id, stmt.value
        return None

    @staticmethod
    def _tcgen05_numbered_name(name: str, base: str) -> bool:
        prefix = f"{base}_"
        return name.startswith(prefix) and name[len(prefix) :].isdecimal()

    @classmethod
    def _tcgen05_grouped_stmt_safe_to_omit(
        cls,
        stmt: ast.stmt,
        *,
        allowed_coord_writes: set[str],
        worklist_metadata: bool,
    ) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr):
            return cls._tcgen05_is_bare_sync_threads_call(stmt.value)
        assignment = cls._tcgen05_single_name_assignment(stmt)
        if assignment is not None:
            name, value = assignment
            if name in allowed_coord_writes or name == "safe_group_id":
                return cls._tcgen05_expr_safe_to_omit(value)
            if name == "group_id":
                return cls._tcgen05_expr_safe_to_omit(value) or (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "load"
                    and not value.args
                    and not value.keywords
                    and cls._tcgen05_expr_safe_to_omit(value.func.value)
                )
            return (
                isinstance(value, ast.Call)
                and cls._tcgen05_call_path(value.func) == "operator.ge"
                and len(value.args) == 2
                and not value.keywords
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id == "group_id"
                and cls._tcgen05_expr_safe_to_omit(value.args[1])
            )
        if (
            not isinstance(stmt, ast.For)
            or not isinstance(stmt.target, ast.Name)
            or not cls._tcgen05_numbered_name(stmt.target.id, "tile_offset")
            or stmt.orelse
            or not cls._tcgen05_expr_safe_to_omit(stmt.iter)
        ):
            return False
        allowed_names = {"acc_copy", "safe_group_id_copy"}
        allowed_bases = {"indices", "mask", *allowed_names}
        if worklist_metadata:
            allowed_names.update({"group_id_copy", "v_0_copy", "v_1_copy"})
            allowed_bases.update(allowed_names)
        for child in stmt.body:
            if isinstance(child, ast.Pass):
                continue
            assignment = cls._tcgen05_single_name_assignment(child)
            if assignment is None:
                return False
            name, value = assignment
            if (
                name != stmt.target.id
                and name not in allowed_names
                and not any(
                    cls._tcgen05_numbered_name(name, base) for base in allowed_bases
                )
            ):
                return False
            if not cls._tcgen05_expr_safe_to_omit(value):
                return False
        return True

    @classmethod
    def _tcgen05_shared_stmt_safe_to_omit(cls, stmt: ast.stmt) -> bool:
        """Return whether a removed shared stmt is dependency-only setup.

        Fully role-local codegen intentionally omits the residual shared
        ``while``. The remaining shared view may still contain scalar
        PID/offset/view setup that role-local loops clone through dependency
        extraction, plus legacy bare ``sync_threads`` barriers that no longer
        bracket shared work after every role has moved out. Other observable
        operations such as copies, pipeline calls, or stores must remain
        rejected so future shared-body work is not silently discarded.
        """
        if isinstance(stmt, ast.Assign):
            return all(
                cls._tcgen05_is_local_assignment_target(t) for t in stmt.targets
            ) and cls._tcgen05_expr_safe_to_omit(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return cls._tcgen05_is_local_assignment_target(stmt.target) and (
                stmt.value is None or cls._tcgen05_expr_safe_to_omit(stmt.value)
            )
        if isinstance(stmt, ast.For):
            return (
                cls._tcgen05_is_local_assignment_target(stmt.target)
                and cls._tcgen05_expr_safe_to_omit(stmt.iter)
                and all(
                    cls._tcgen05_shared_stmt_safe_to_omit(child) for child in stmt.body
                )
                and all(
                    cls._tcgen05_shared_stmt_safe_to_omit(child)
                    for child in stmt.orelse
                )
            )
        if isinstance(stmt, ast.If):
            return (
                cls._tcgen05_expr_safe_to_omit(stmt.test)
                and all(
                    cls._tcgen05_shared_stmt_safe_to_omit(child) for child in stmt.body
                )
                and all(
                    cls._tcgen05_shared_stmt_safe_to_omit(child)
                    for child in stmt.orelse
                )
            )
        if isinstance(stmt, ast.Expr):
            return cls._tcgen05_is_bare_sync_threads_call(stmt.value)
        return isinstance(stmt, ast.Pass)

    def _assert_tcgen05_omit_shared_loop_safe(
        self,
        partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody,
        post_loop_stmts: list[ast.stmt] | None = None,
    ) -> None:
        unsafe = self._tcgen05_unsafe_shared_stmts(partition)
        assert not unsafe, (
            "tcgen05 fully role-local codegen would discard observable shared "
            "statement(s) while omitting the residual shared loop: "
            + "; ".join(ast.unparse(stmt) for stmt in unsafe)
        )
        dependencies = self._tcgen05_shared_post_loop_dependencies(
            partition, post_loop_stmts
        )
        assert not dependencies, (
            "tcgen05 fully role-local codegen would discard shared definition(s) "
            "used by post-loop cleanup: " + ", ".join(sorted(dependencies))
        )

    def _tcgen05_unsafe_shared_stmts(
        self, partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody
    ) -> list[ast.stmt]:
        return [
            stmt
            for stmt in partition.shared_body_extracted
            if not self._tcgen05_shared_stmt_safe_to_omit(stmt)
        ]

    def _tcgen05_shared_post_loop_dependencies(
        self,
        partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody,
        post_loop_stmts: list[ast.stmt] | None,
    ) -> set[str]:
        shared_writes: set[str] = set()
        for stmt in partition.shared_body_extracted:
            _, writes = _stmt_name_uses(stmt)
            shared_writes.update(writes)
        post_loop_reads: set[str] = set()
        for stmt in post_loop_stmts or ():
            reads, _ = _stmt_name_uses(stmt)
            post_loop_reads.update(reads)
        return shared_writes & post_loop_reads

    def _tcgen05_shared_loop_has_meaningful_work(
        self,
        partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody,
        post_loop_stmts: list[ast.stmt],
    ) -> bool:
        """Return whether the residual shared loop must be emitted.

        This is deliberately fail-closed: any statement outside the narrow
        side-effect-free allowlist, or any definition consumed by post-loop
        cleanup, makes the shared loop meaningful. Kernel-family admission
        only establishes that independent role schedulers are available; the
        actual residual body decides whether codegen may omit the loop.
        """
        unsafe = (
            self._tcgen05_grouped_unsafe_shared_stmts(partition)
            if self._tcgen05_uses_grouped_static_persistent()
            else self._tcgen05_unsafe_shared_stmts(partition)
        )
        return bool(
            unsafe
            or self._tcgen05_shared_post_loop_dependencies(partition, post_loop_stmts)
        )

    def _assert_tcgen05_grouped_omit_shared_loop_safe(
        self, partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody
    ) -> None:
        unsafe = self._tcgen05_grouped_unsafe_shared_stmts(partition)
        assert not unsafe, (
            "tcgen05 grouped static scheduler would discard observable shared "
            "statement(s) while omitting the residual shared loop: "
            + "; ".join(ast.unparse(stmt) for stmt in unsafe)
        )

    def _tcgen05_grouped_unsafe_shared_stmts(
        self, partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody
    ) -> list[ast.stmt]:
        plan = self._tcgen05_plan()
        device_split_sizes = bool(
            plan is not None
            and plan.grouped is not None
            and plan.grouped.device_split_sizes
        )
        if device_split_sizes:
            return self._tcgen05_unsafe_shared_stmts(partition)
        worklist_metadata = bool(
            plan is not None
            and plan.grouped is not None
            and plan.grouped.real_groups is not None
        )
        allowed_coord_writes = {
            "virtual_pid",
            "pid_0",
            "pid_1",
            "tile_offset_0",
            "tile_offset_1",
        }
        if worklist_metadata:
            # Generated segment worklists re-express the original
            # parser-order work row as the grouped scheduler's pseudo-group.
            # Once the scheduler metadata statements are injected, the old
            # segment-loop coordinate and scalar scaffolding is dependency-only.
            allowed_coord_writes.update(
                {
                    "pid_2",
                    "tile_offset_2",
                    "tile_offset_3",
                    "indices_2",
                    "indices_3",
                    "mask_2",
                    "mask_3",
                }
            )
        return [
            stmt
            for stmt in partition.shared_body_extracted
            if not self._tcgen05_grouped_stmt_safe_to_omit(
                stmt,
                allowed_coord_writes=allowed_coord_writes,
                worklist_metadata=worklist_metadata,
            )
        ]

    def _build_tcgen05_persistent_tile_body_role_local(
        self,
        device_function: DeviceFunction,
        layout: _Tcgen05PersistentLayout,
        partition: Tcgen05PersistentProgramIDs._PartitionedRoleBody,
        *,
        build_shared_tile_body: bool = True,
        epi_role_prelude_stmts: list[ast.stmt] | None = None,
        post_loop_stmts: list[ast.stmt] | None = None,
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """Build the per-tile body in role-local-while form.

        Returns ``(role_local_whiles, shared_tile_body)`` where:

        - ``role_local_whiles`` is a list of role-local ``while`` siblings
          -- one per unique ``role_predicate`` in
          ``partition.role_blocks_extracted``. Multiple extracted role
          blocks sharing the same predicate are merged into a single
          role-local loop/body with their statements concatenated in the
          order they appear in the source body, so per-tile ordering
          across the role's statements is preserved (otherwise tile 0's
          first chunk would run for every tile before tile 0's second
          chunk ran, breaking the AB-pipeline ordering). Each loop is
          wrapped in ``if {role_predicate}:`` so only the matching
          warps enter.
        - ``shared_tile_body`` is the optional per-tile body for the shared
          ``while`` (the work-tile body without the extracted role blocks).
          Built via :meth:`_build_tcgen05_persistent_tile_body` with existing
          ``cute.arch.sync_threads()`` calls preserved. The caller omits this
          loop only when the residual statements are dependency-only setup or
          legacy barriers that no longer protect shared work.

        Caller wires both into the persistent kernel as siblings of
        each other inside the same setup list when the residual shared loop
        is needed. Each role-local ``while`` runs only on its predicated warps.

        **Current limitation.** The TMA-load, MMA-exec, and TMA-store
        epilogue roles are extracted today. Single-root static full-tile
        multi-tile correctness is validated for ``cluster_m == 1`` and for
        role-local CtaGroup.TWO ``cluster_m == 2`` up to the validated K-tile
        cap, using role-local scheduler loops over the capped persistent grid.
        Partial fallback shapes, CtaGroup.TWO shapes above the K-tile cap, and
        multi-root ForEach kernels remain guarded for runtime execution, and
        autotune keeps cluster_m=2 out of the search until the G3 ownership path
        is benchmarked.
        """
        if build_shared_tile_body:
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
        else:
            assert post_loop_stmts is not None, (
                "omitting the tcgen05 shared loop requires explicit post-loop "
                "dependency validation"
            )
            if self._tcgen05_uses_grouped_static_persistent():
                self._assert_tcgen05_grouped_omit_shared_loop_safe(partition)
                dependencies = self._tcgen05_shared_post_loop_dependencies(
                    partition, post_loop_stmts
                )
                assert not dependencies, (
                    "tcgen05 grouped static scheduler would discard shared "
                    "definition(s) used by post-loop cleanup: "
                    + ", ".join(sorted(dependencies))
                )
            else:
                self._assert_tcgen05_omit_shared_loop_safe(partition, post_loop_stmts)
            shared_tile_body = []
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
        cute_state = device_function.cute_state
        use_full_edge_scheduler_split = (
            self._tcgen05_has_scheduler_warp()
            and cute_state.has_tcgen05_epi_role_full_edge_split
        )
        # Cycle-94 merge gate: the store warp is the aux residual producer when
        # there is a store warp, NO C-input warp, and a single-store-value
        # exact-shape aux ring exists. In that case the aux GMEM->SMEM producer
        # body is injected into the (widened) epilogue role-local while on the
        # store warp rather than emitted as a standalone C-input role-local
        # while (which would be a second per-warp sched consumer). The standalone
        # C-input while below stays gated on ``has_c_input_warp`` and does not
        # fire in the merge.
        #
        # TMA-ONLY: the merge injects a TMA bulk producer body; there is no SIMT
        # store-warp producer. ``cute_mma._emit_mma_pipeline`` only allocates the
        # aux pipeline for the store warp when ``aux_load_mode=tma``, so for
        # ``store_warps=1 + SIMT aux`` the pipeline plan is absent and the gate
        # closes (the kernel falls back to direct-GMEM aux). The
        # ``use_tma_load`` check below makes that requirement explicit and
        # defends against a future SIMT-producing aux plan reaching this path.
        store_merge_plan = self._tcgen05_plan()
        aux_plan_for_merge = device_function.cute_state.aux_pipeline_plan
        store_aux_merge_active = (
            store_merge_plan is not None
            and store_merge_plan.has_store_warp
            and not store_merge_plan.has_c_input_warp
            and bool(store_merge_plan.c_input_aux_tensor_descriptors)
            and len(
                {
                    d.store_value_node
                    for d in store_merge_plan.c_input_aux_tensor_descriptors
                }
            )
            <= 1
            and aux_plan_for_merge is not None
            and aux_plan_for_merge.use_tma_load
        )
        for i, predicate in enumerate(ordered_predicates):
            stmts = merged[predicate]
            split_epi_role = (
                use_full_edge_scheduler_split
                and predicate == self._tcgen05_epi_role_predicate()
            )
            if split_epi_role:
                unclassified = [
                    stmt
                    for stmt in stmts
                    if not cute_state.is_tcgen05_epi_role_full_tile(stmt)
                    and not cute_state.is_tcgen05_epi_role_edge_tile(stmt)
                ]
                assert not unclassified, (
                    "scheduler full/edge split found unclassified epilogue "
                    "role statement(s): "
                    + "; ".join(ast.unparse(stmt) for stmt in unclassified)
                )

            # Under a scheduler full/edge split, non-epi roles keep the same
            # body for both phases so they consume both scheduler streams in
            # order; only the epi role swaps in phase-specific store bodies.
            phase_names = ("full", "edge") if use_full_edge_scheduler_split else ("",)
            for phase in phase_names:
                if not split_epi_role:
                    current_stmts = stmts
                elif phase == "full":
                    current_stmts = [
                        stmt
                        for stmt in stmts
                        if cute_state.is_tcgen05_epi_role_full_tile(stmt)
                    ]
                else:
                    current_stmts = [
                        stmt
                        for stmt in stmts
                        if cute_state.is_tcgen05_epi_role_edge_tile(stmt)
                    ]
                if not current_stmts:
                    continue
                if use_full_edge_scheduler_split:
                    # Each phase needs a distinct AST body; reusing nodes would
                    # let later dependency extraction mutate shared structure.
                    current_stmts = [_clone_stmt(stmt) for stmt in current_stmts]
                merged_block = self._PersistentRoleBlock(
                    role_predicate=predicate, stmts=current_stmts
                )
                dependency_stmts = self._role_local_dependency_stmts(
                    partition.shared_body_extracted, current_stmts
                )
                if use_full_edge_scheduler_split:
                    dependency_stmts = [_clone_stmt(stmt) for stmt in dependency_stmts]
                role_prelude_stmts: list[ast.stmt] | None = None
                if (
                    predicate == self._tcgen05_epi_role_predicate()
                    and phase != "edge"
                    and epi_role_prelude_stmts
                ):
                    role_prelude_stmts = [
                        _clone_stmt(stmt) for stmt in epi_role_prelude_stmts
                    ]
                suffix = f"_{phase}" if phase else ""
                # Cycle-94 merge: build the store warp's aux producer body for
                # injection into the epilogue role-local while. Only on the epi
                # predicate, and (under a full/edge split) only the full-tile
                # phase stages the aux ring — the edge phase uses SIMT direct
                # GMEM aux, like the standalone C-input builder.
                store_aux_per_tile_stmts: list[ast.stmt] | None = None
                store_aux_predicate: str | None = None
                if (
                    store_aux_merge_active
                    and predicate == self._tcgen05_epi_role_predicate()
                    and phase != "edge"
                ):
                    aux_inline = self._build_c_input_warp_role_local_while(
                        device_function,
                        layout,
                        shared_body_extracted=partition.shared_body_extracted,
                        tile_phase="all",
                        inline_aux_only=True,
                    )
                    assert isinstance(aux_inline, list)
                    store_aux_per_tile_stmts = aux_inline
                    assert store_merge_plan is not None
                    store_aux_predicate = (
                        "cute.arch.make_warp_uniform(cute.arch.warp_idx()) "
                        f"== cutlass.Int32({store_merge_plan.store_warp_id})"
                    )
                role_local_whiles.append(
                    self._build_role_local_while(
                        device_function,
                        layout,
                        merged_block,
                        scheduler_var_prefix=f"tcgen05_role_local_{i}{suffix}",
                        dependency_stmts=dependency_stmts,
                        role_prelude_stmts=role_prelude_stmts,
                        emit_pdl_wait=phase != "edge",
                        initialize_tile_counter=phase != "edge",
                        store_aux_per_tile_stmts=store_aux_per_tile_stmts,
                        store_aux_predicate=store_aux_predicate,
                    )
                )
        # ``ROLE_LOCAL_WITH_SCHEDULER`` adds a fourth role-local while
        # for the dedicated scheduler warp. Its body is constructed
        # in-place (no source statements to extract from device IR).
        # Append after the consumer roles so the scheduler-warp
        # loop sits at the end of the per-tile setup; the
        # producer/consumer pipeline pairing is order-independent
        # because the consumers wait on a barrier the scheduler
        # arms.
        if self._tcgen05_has_scheduler_warp():
            role_local_whiles.append(
                self._build_scheduler_warp_role_local_while(device_function, layout)
            )
        # ``c_input_warp_count > 0`` AND a non-empty
        # ``aux_tensor_descriptors`` adds a fifth role-local while for
        # the C-input warp (``cute_plan.md`` §7.5.3.2 producer-body
        # split). Post-cycle-2a the role body has the
        # consumer_wait / valid-read / release machinery so the
        # warp participates as a sched-pipeline consumer and
        # publishes ``virtual_pid_var`` for cycle 2b. The aux
        # pipeline storage (SMEM ring + ``c_pipeline_aux``) is
        # allocated alongside but the
        # producer/consumer barrier handshake stays dormant until
        # cycle 2b (producer side + cooperative copy) and cycle 3
        # (consumer-side SMEM read flip) land together. Gating on
        # the descriptors being non-empty preserves byte identity
        # for every ``c_input_warps=0`` config (today's default)
        # and for ``c_input_warps=1`` configs that don't have a
        # residual epilogue (the walker returns an empty tuple in
        # that case).
        plan = self._tcgen05_plan()
        # Multi-store fan-out guard mirrors the same check in
        # ``cute_mma._emit_mma_pipeline``: the productive body
        # only emits when every aux descriptor for this matmul
        # comes from a single ``store_value_node``. The
        # cute_mma path uses the same predicate to gate the
        # ``c_pipeline_aux`` allocation; both must agree or the
        # role-local while will try to access a missing
        # pipeline plan.
        if (
            plan is not None
            and plan.has_c_input_warp
            and plan.c_input_aux_tensor_descriptors
            and len({d.store_value_node for d in plan.c_input_aux_tensor_descriptors})
            <= 1
        ):
            c_input_phases = (
                ("full", "edge") if use_full_edge_scheduler_split else ("all",)
            )
            for c_input_phase in c_input_phases:
                c_input_while = self._build_c_input_warp_role_local_while(
                    device_function,
                    layout,
                    # ``shared_body_extracted`` carries the post-PID-
                    # decomposition statements (incl. the L2-grouping
                    # ``pid_0`` / ``pid_1`` / ``tile_offset_0`` /
                    # ``tile_offset_1`` chain). The C-input producer
                    # body needs the same chain so its per-CTA aux
                    # GMEM tile coords match the consumer's
                    # post-L2-remapped tile coords — without this
                    # the producer fetches a misaligned aux tile
                    # under ``l2_groupings=[g>1]`` and the residual
                    # add reads wrong rows / columns of the
                    # auxiliary tensor (cycle 2i: 60-69% mismatched
                    # elements vs eager).
                    shared_body_extracted=partition.shared_body_extracted,
                    tile_phase=c_input_phase,
                )
                # ``inline_aux_only`` is False here, so the builder returns the
                # full role-local while statement (not the merge tuple).
                assert isinstance(c_input_while, ast.stmt)
                role_local_whiles.append(c_input_while)
        return role_local_whiles, shared_tile_body

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        return self._setup_tcgen05_persistent_kernel(device_function)
