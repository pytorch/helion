from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .. import exc
from ..autotuner.config_spec import CROSS_LOOP_SCHEDULE_BARRIER
from ..autotuner.config_spec import CROSS_LOOP_SCHEDULE_CONFIG
from ..autotuner.config_spec import CROSS_LOOP_SCHEDULE_DEFAULT
from ..autotuner.config_spec import CROSS_LOOP_SCHEDULE_STATIC_PIPELINE
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cross_loop_scheduler import CountedEventPlan
from .cross_loop_scheduler import EventContribution
from .cross_loop_scheduler import EventUse
from .cross_loop_scheduler import build_cross_loop_schedule
from .device_function import TensorArg
from .host_function import HostFunction
from .program_id import _clone_ast_value
from .program_id import _clone_stmt
from .program_id import typed_program_id
from .tile_dependency import TILE_DEPENDENCY_SCOPE_ID_ATTR
from .tile_dependency import LogicalDomain
from .tile_dependency import LogicalRelation
from .tile_dependency import instantiate_logical_domains
from .tile_dependency import logical_axis_symbol
from .tile_dependency import nested_logical_axes
from .tile_dependency import physical_traversal_relation
from .tile_dependency import tile_dependency_scope_id
from .tile_strategy import L2GroupingProgramIDs

if TYPE_CHECKING:
    from collections.abc import Callable

    from .device_function import DeviceFunction
    from .program_id import ForEachProgramID
    from .program_id import PersistentProgramIDs
    from .program_id import PIDInfo
    from .program_id import ProgramIDs


# Independent keyed-event counters occupy distinct cache lines so polling one
# dependency does not contend with publication to another. Cross-loop event
# state is currently CUDA-only, where 128 bytes is a conservative L2 line
# alignment. The layout is expressed in bytes rather than a model-shaped
# counter count.
_CROSS_LOOP_COUNTER_ALIGNMENT_BYTES = 128
_CROSS_LOOP_COUNTER_DTYPE = torch.uint32
_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS = (
    _CROSS_LOOP_COUNTER_ALIGNMENT_BYTES // _CROSS_LOOP_COUNTER_DTYPE.itemsize
)


def _ast_fingerprint(nodes: list[ast.stmt]) -> tuple[str, ...]:
    """Return a location-independent fingerprint for an opaque computation body."""
    return tuple(ast.dump(node, include_attributes=False) for node in nodes)


def _clone_opaque_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Clone a tile body while proving that no computation was rewritten."""
    cloned = [_clone_stmt(statement) for statement in body]
    if _ast_fingerprint(cloned) != _ast_fingerprint(body):
        raise AssertionError("opaque tile-body cloning changed its computation")
    return cloned


def _clone_opaque_statements_with_loop_rewrite(
    body: list[ast.stmt],
    rewrite: Callable[[ast.For], list[ast.stmt] | None],
) -> list[ast.stmt]:
    """Clone an opaque body while replacing selected loops."""

    def clone(value: object) -> object:
        if isinstance(value, list):
            result: list[object] = []
            for item in value:
                if (
                    isinstance(item, ast.For)
                    and (replacement := rewrite(item)) is not None
                ):
                    result.extend(replacement)
                else:
                    result.append(clone(item))
            return result
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, ast.AST):
            fields = {field: clone(getattr(value, field)) for field in value._fields}
            if isinstance(value, ExtendedAST):
                cloned = value.copy(**fields)
            else:
                cloned = ast.copy_location(type(value)(**fields), value)
            if (
                scope_id := getattr(value, TILE_DEPENDENCY_SCOPE_ID_ATTR, None)
            ) is not None:
                setattr(cloned, TILE_DEPENDENCY_SCOPE_ID_ATTR, scope_id)
            return cloned
        return value

    return cast("list[ast.stmt]", clone(body))


def _clone_opaque_loop_segment(
    loop: ast.For,
    *,
    begin: ast.expr | None = None,
    end: ast.expr | None = None,
) -> ast.For:
    """Clone one existing loop, changing only scheduling range boundaries."""
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


def _clone_opaque_statements_with_scope_stages(
    body: list[ast.stmt],
    *,
    scope_id: int,
    split_iteration_offsets: tuple[int, ...],
    stage_waits: tuple[tuple[ast.stmt, ...], ...],
) -> list[ast.stmt]:
    """Split one stable DeviceIR scope loop and wait before each segment."""
    if len(stage_waits) != len(split_iteration_offsets) + 1:
        raise AssertionError("each scope-loop segment requires one wait")
    scheduled = False

    def rewrite(loop: ast.For) -> list[ast.stmt] | None:
        nonlocal scheduled
        if tile_dependency_scope_id(loop) != scope_id:
            return None
        if scheduled:
            raise AssertionError("one dependency scope must identify one lowered loop")
        if not isinstance(loop.iter, ast.Call) or len(loop.iter.args) < 2:
            raise AssertionError("nested scope scheduling requires a range-like loop")
        begin = ast.unparse(loop.iter.args[0])
        step = ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
        split_offsets = tuple(
            f"({begin}) + ({offset}) * ({step})" for offset in split_iteration_offsets
        )
        boundaries = (None, *split_offsets, None)
        result: list[ast.stmt] = []
        for index, waits in enumerate(stage_waits):
            result.extend(_clone_opaque_statements(list(waits)))
            begin_text = boundaries[index]
            end_text = boundaries[index + 1]
            segment_begin = (
                cast("ast.expr", expr_from_string(begin_text))
                if begin_text is not None
                else None
            )
            segment_end = (
                cast("ast.expr", expr_from_string(end_text))
                if end_text is not None
                else None
            )
            result.append(
                _clone_opaque_loop_segment(
                    loop,
                    begin=segment_begin,
                    end=segment_end,
                )
            )
        scheduled = True
        return result

    cloned = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
    if not scheduled:
        present_scope_ids = sorted(
            found_scope_id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.For)
            if (found_scope_id := tile_dependency_scope_id(node)) is not None
        )
        raise AssertionError(
            f"missing dependency scope {scope_id}; found {present_scope_ids}"
        )
    return cloned


def _stage_root_ranges(owner: ForEachProgramID) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    begin = 0
    for index in range(1, len(owner.case_phases) + 1):
        if (
            index == len(owner.case_phases)
            or owner.case_phases[index] != owner.case_phases[index - 1]
        ):
            result.append((begin, index))
            begin = index
    return result


def _extract_case_bodies(
    owner: ForEachProgramID,
    base_body: list[ast.stmt],
) -> list[list[ast.stmt]]:
    if len(owner.cases) == 1:
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
    assert len(result) == len(owner.cases)
    return result


def _case_pid_info(case: ProgramIDs) -> list[PIDInfo]:
    if isinstance(case, L2GroupingProgramIDs):
        assert case.parent_strategy is not None
        return case.parent_strategy.pid_info
    return case.pid_info


def _static_case_axes(
    owner: ForEachProgramID,
    root: int,
    device_function: DeviceFunction,
) -> list[tuple[int, int]] | None:
    env = CompileEnvironment.current()
    task_families = HostFunction.current().device_ir.task_families
    if root >= len(task_families):
        return None
    task_family = task_families[root]
    result: list[tuple[int, int]] = []
    for info in _case_pid_info(owner.cases[root]):
        logical_axis = task_family.axis(info.block_id)
        if logical_axis is None:
            return None
        numel_expr = logical_axis.extent
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


def _static_case_geometry(
    owner: ForEachProgramID,
    root: int,
    device_function: DeviceFunction,
) -> tuple[tuple[int, ...], dict[int, int], dict[int, int]] | None:
    axes = _static_case_axes(owner, root, device_function)
    if axes is None:
        return None
    infos = _case_pid_info(owner.cases[root])
    axis_order = tuple(info.block_id for info in infos)
    axis_counts = {
        info.block_id: (numel + block - 1) // block
        for info, (numel, block) in zip(infos, axes, strict=True)
    }
    block_sizes = {
        info.block_id: block for info, (_, block) in zip(infos, axes, strict=True)
    }
    return axis_order, axis_counts, block_sizes


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


def _effective_l2_group_size(
    case: ProgramIDs,
    axis_order: tuple[int, ...],
    axis_counts: dict[int, int],
) -> int | None:
    """Return the nontrivial L2 grouping applied by one root traversal."""
    if not isinstance(case, L2GroupingProgramIDs) or len(axis_order) < 2:
        return None
    first_axis, second_axis = axis_order[:2]
    if axis_counts[second_axis] == 1 or case.group_size >= axis_counts[first_axis]:
        return None
    return case.group_size


def _root_physical_traversals(
    owner: ForEachProgramID,
    root_domains: tuple[LogicalDomain, ...],
    case_geometries: tuple[tuple[tuple[int, ...], dict[int, int], dict[int, int]], ...],
) -> tuple[LogicalRelation, ...] | None:
    """Bind each logical root domain to its configured PID traversal."""
    if len(root_domains) != len(case_geometries):
        return None
    result: list[LogicalRelation] = []
    for root, (domain, geometry) in enumerate(
        zip(root_domains, case_geometries, strict=True)
    ):
        physical_axis_order, axis_counts, block_sizes = geometry
        if (
            set(domain.axis_order) != set(physical_axis_order)
            or domain.axis_counts
            != {axis: axis_counts[axis] for axis in domain.axis_order}
            or domain.block_sizes
            != {axis: block_sizes[axis] for axis in domain.axis_order}
        ):
            return None
        case = owner.cases[root]
        l2_group_size = _effective_l2_group_size(
            case,
            physical_axis_order,
            axis_counts,
        )
        result.append(
            physical_traversal_relation(
                domain,
                physical_axis_order,
                l2_group_size=l2_group_size,
            )
        )
    return tuple(result)


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


def _wait_for_dependencies(
    *,
    device_function: DeviceFunction,
    dependencies: tuple[tuple[str, str], ...],
    prefix: str,
) -> list[ast.stmt]:
    """Emit every acquire wait in one graph-derived dependency set."""
    return [
        statement
        for counter, target in dependencies
        for statement in _wait_for_counter(
            device_function=device_function,
            counter=counter,
            target=target,
            prefix=prefix,
        )
    ]


def _emit_counted_event_on_ready(
    *,
    counter: str,
    epoch: str,
    expected_arrivals: int,
    previous: str,
    on_ready: list[ast.stmt],
) -> list[ast.stmt]:
    """Contribute once and run ``on_ready`` for the final arrival."""
    return [
        statement_from_string(
            f"{previous} = tl.atomic_add({counter}, 1, sem='acq_rel', scope='gpu')"
        ),
        create(
            ast.If,
            test=expr_from_string(
                f"{previous} == tl.cast({epoch}, tl.uint32) * "
                f"tl.cast({expected_arrivals}, tl.uint32) - 1"
            ),
            body=on_ready,
            orelse=[],
        ),
    ]


def _publication_barrier(device_function: DeviceFunction) -> ast.stmt:
    if cast("int", device_function.config.get("num_warps", 1)) != 1:
        return statement_from_string("tl.debug_barrier()")
    sync = device_function.new_var("tile_dependency_publication_sync", dce=False)
    return statement_from_string(
        f"{sync} = tl.inline_asm_elementwise("
        "asm='bar.warp.sync 0xffffffff; mov.u32 $0, $1;', "
        "constraints='=r,r', args=[tl.arange(0, 32)], "
        "dtype=tl.uint32, is_pure=False, pack=1)"
    )


def _register_cross_loop_state(
    device_function: DeviceFunction,
    *,
    name_hint: str,
    numel: str,
    dtype: torch.dtype,
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
        (like.host_str(), numel, str(dtype))
    )
    return name


def _outline_cross_loop_region(
    device_function: DeviceFunction,
    *,
    name_hint: str,
    body: list[ast.stmt],
    extra_argument_names: tuple[str, ...] = (),
    noinline: bool = False,
) -> ast.stmt:
    """Outline a scheduled region while keeping its computation opaque."""
    helper_name, arguments = device_function.register_triton_outlined_helper(
        name_hint,
        body,
        extra_argument_names=extra_argument_names,
        noinline=noinline,
    )
    return statement_from_string(f"{helper_name}({', '.join(arguments)})")


def _outline_opaque_tile_body(
    owner: ForEachProgramID,
    device_function: DeviceFunction,
    *,
    root: int,
    logical_pid: str,
    body: list[ast.stmt],
    name_suffix: str = "",
    extra_argument_names: tuple[str, ...] = (),
    noinline: bool = False,
) -> ast.stmt:
    """Create a call containing exactly one original tile body."""
    suffix = f"_{name_suffix}" if name_suffix else ""
    return _outline_cross_loop_region(
        device_function,
        name_hint=f"tile_dependency_root_{root}{suffix}",
        body=[
            statement_from_string(f"{owner.shared_pid_var} = {logical_pid}"),
            *_clone_opaque_statements(body),
        ],
        extra_argument_names=extra_argument_names,
        noinline=noinline,
    )


def emit_cross_loop_schedule(
    owner: ForEachProgramID,
    strategy: PersistentProgramIDs,
    device_function: DeviceFunction,
    total_expr: str,
) -> list[ast.stmt]:
    """Emit monotonic arrival-counter phases for a persistent launch.

    Workers publish release arrivals and consumers acquire-poll the
    corresponding counters. The targets are epoch-scaled, so fixed CUDA
    Graph arguments need neither a reset kernel nor a host-side epoch update.
    """
    schedule = device_function.config.get(
        CROSS_LOOP_SCHEDULE_CONFIG,
        CROSS_LOOP_SCHEDULE_DEFAULT,
    )
    if schedule == CROSS_LOOP_SCHEDULE_BARRIER:
        CompileEnvironment.current().has_barrier = True
        device_function.triton_minimum_resident_programs = strategy.grid_size_expr
        return owner._emit_phase_loops(strategy, device_function, total_expr)
    if schedule != CROSS_LOOP_SCHEDULE_STATIC_PIPELINE:
        raise exc.InvalidConfig(
            f"unknown {CROSS_LOOP_SCHEDULE_CONFIG} value {schedule!r}"
        )

    configured_case_geometries = tuple(
        _static_case_geometry(owner, root, device_function)
        for root in range(len(owner.cases))
    )
    if any(geometry is None for geometry in configured_case_geometries):
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_SCHEDULE_CONFIG}="
            f"{CROSS_LOOP_SCHEDULE_STATIC_PIPELINE!r} requires concrete "
            "top-level task counts"
        )
    case_geometries = tuple(
        geometry for geometry in configured_case_geometries if geometry is not None
    )
    worker = typed_program_id(0)
    epoch_var = device_function.new_var("tile_dependency_epoch", dce=False)
    base_body = owner._prepare_persistent_body(
        cast("list[ast.stmt]", device_function.body),
        device_function,
        strategy.virtual_pid_var,
    )
    case_bodies = _extract_case_bodies(owner, base_body)
    opaque_case_fingerprints = tuple(_ast_fingerprint(body) for body in case_bodies)
    dependency_plan = HostFunction.current().device_ir.tile_dependency_graph
    assert dependency_plan is not None
    indexing = device_function.config.get("indexing", ())

    def uses_tensor_descriptor(memory_op_index: int) -> bool:
        if isinstance(indexing, str):
            return indexing == "tensor_descriptor"
        return (
            isinstance(indexing, (list, tuple))
            and 0 <= memory_op_index < len(indexing)
            and indexing[memory_op_index] == "tensor_descriptor"
        )

    unpublishable_scope_ids = frozenset(
        scope_id
        for access in dependency_plan.accesses
        if access.kind == "store" and uses_tensor_descriptor(access.memory_op_index)
        for scope_id in dependency_plan.scope_ids_by_access[access.access_id]
    )
    publishable_scope_ids = (
        frozenset(
            scope.scope_id
            for scope in dependency_plan.execution_scopes
            if not scope.is_root and scope.scope_id not in unpublishable_scope_ids
        )
        if unpublishable_scope_ids
        else None
    )
    configured_worker_count = CompileEnvironment.current().config_spec.num_sm * cast(
        "int", device_function.config.get("num_sm_multiplier", 1)
    )
    root_axis_geometry: dict[int, tuple[int, int]] = {}
    for _axis_order, axis_counts, block_sizes in case_geometries:
        for block_id, task_count in axis_counts.items():
            geometry = (task_count, block_sizes[block_id])
            previous = root_axis_geometry.setdefault(block_id, geometry)
            if previous != geometry:
                raise AssertionError(
                    f"inconsistent configured geometry for block axis {block_id}"
                )
    axis_geometry: dict[int, tuple[int, int]] = {}
    for block_id in range(len(CompileEnvironment.current().block_sizes)):
        geometry = root_axis_geometry.get(block_id)
        if geometry is None:
            geometry = _static_block_axis_geometry(block_id, device_function)
        if geometry is not None:
            axis_geometry[block_id] = geometry
    configured_root_domains, scope_domains = instantiate_logical_domains(
        dependency_plan,
        axis_geometry=axis_geometry,
    )
    if any(domain is None for domain in configured_root_domains):
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_SCHEDULE_CONFIG}="
            f"{CROSS_LOOP_SCHEDULE_STATIC_PIPELINE!r} requires static "
            "root domains"
        )
    root_domains = tuple(
        domain for domain in configured_root_domains if domain is not None
    )
    case_offsets: list[int] = []
    running_offset = 0
    for domain in root_domains:
        case_offsets.append(running_offset)
        running_offset += domain.size
    root_traversals = _root_physical_traversals(
        owner,
        root_domains,
        case_geometries,
    )
    if root_traversals is None:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_SCHEDULE_CONFIG}="
            f"{CROSS_LOOP_SCHEDULE_STATIC_PIPELINE!r} requires a "
            "representable root traversal"
        )
    cross_loop_schedule = build_cross_loop_schedule(
        dependency_plan=dependency_plan,
        root_traversals=root_traversals,
        scope_domains=scope_domains,
        worker_count=configured_worker_count,
        publishable_scope_ids=publishable_scope_ids,
    )
    root_completion_edges = cross_loop_schedule.root_completion_edges
    all_counted_event_plans = cross_loop_schedule.counted_events
    nested_scope_event_plans = tuple(
        plan
        for plan in all_counted_event_plans
        if any(use.consumer_scope_id is not None for use in plan.uses)
    )
    counted_event_plans = tuple(
        plan
        for plan in all_counted_event_plans
        if all(use.consumer_scope_id is None for use in plan.uses)
    )
    launch_worker_count = cross_loop_schedule.worker_schedule.worker_count

    active_worker_counts_by_root = {
        root: len(cross_loop_schedule.worker_schedule.workers_for_root(root))
        for root in range(len(root_domains))
    }
    local_task_count_by_root: dict[int, int] = {}
    for plan in counted_event_plans:
        use = plan.local_use
        if use is None:
            continue
        if not use.keys.is_total_function():
            raise AssertionError("a local event must cover its complete root")
        local_task_count_by_root[use.consumer_root] = use.keys.source_domain.size

    def root_completion_arrival_count(root: int) -> int:
        return active_worker_counts_by_root[root] + local_task_count_by_root.get(
            root, 0
        )

    # Reject grids that cannot residently fit when the device is otherwise
    # idle. Concurrent-stream residency remains an explicit unresolved
    # contract for this non-cooperative static lowering.
    device_function.triton_minimum_resident_programs = strategy.grid_size_expr
    device_function.preamble.extend(strategy._persistent_setup_statements(total_expr))
    counted_event_offsets: dict[CountedEventPlan, int] = {}
    counted_event_counter_count = 0
    counted_event_key_stride = _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    for plan in all_counted_event_plans:
        counted_event_offsets[plan] = counted_event_counter_count
        counted_event_counter_count += plan.key_count * counted_event_key_stride
    root_completion_producer_roots = sorted(
        {producer for producer, _consumer in root_completion_edges}
    )
    root_completion_indices = {
        root: index for index, root in enumerate(root_completion_producer_roots)
    }
    state_count = 0

    def reserve_state(count: int) -> int | None:
        nonlocal state_count
        if not count:
            return None
        state_count = (
            (state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
            // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
            * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        )
        offset = state_count
        state_count += count
        return offset

    counted_event_state_offset = reserve_state(counted_event_counter_count)
    root_completion_state_offset = reserve_state(
        len(root_completion_producer_roots) * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    static_state_base = str(
        (launch_worker_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
        // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    state_arg = _register_cross_loop_state(
        device_function,
        name_hint="tile_dependency_state",
        numel=f"{static_state_base} + {state_count}",
        dtype=_CROSS_LOOP_COUNTER_DTYPE,
    )

    def state_section(offset: int | None) -> str | None:
        if offset is None:
            return None
        return f"{state_arg} + ({static_state_base}) + {offset}"

    epoch_arg = state_arg
    counted_event_arg = state_section(counted_event_state_offset)
    root_completion_counter_arg = state_section(root_completion_state_offset)

    stage_root_ranges = _stage_root_ranges(owner)
    result: list[ast.stmt] = [
        statement_from_string(f"{epoch_var} = tl.load({epoch_arg} + {worker}) + 1")
    ]
    consumed_on_ready_roots = {
        local_use.consumer_root
        for plan in counted_event_plans
        if (local_use := plan.local_use) is not None
    }

    root_completion_incoming: dict[int, tuple[int, ...]] = {
        consumer: tuple(
            sorted(
                producer
                for producer, target in root_completion_edges
                if target == consumer
            )
        )
        for consumer in {consumer for _producer, consumer in root_completion_edges}
    }

    def root_completion_counter(root: int) -> str:
        assert root_completion_counter_arg is not None
        return (
            f"{root_completion_counter_arg} + "
            f"{root_completion_indices[root] * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}"
        )

    def root_completion_dependency(root: int) -> tuple[str, str]:
        arrivals = root_completion_arrival_count(root)
        return (
            root_completion_counter(root),
            f"tl.cast({epoch_var}, tl.uint32) * tl.cast({arrivals}, tl.uint32)",
        )

    def root_completion_input_dependencies(
        root: int,
    ) -> tuple[tuple[str, str], ...]:
        producers = root_completion_incoming.get(root, ())
        return tuple(root_completion_dependency(producer) for producer in producers)

    def root_completion_publication(root: int) -> list[ast.stmt]:
        if root not in root_completion_indices:
            return []
        completion_counter = root_completion_counter(root)
        arrivals = root_completion_arrival_count(root)
        result = [_publication_barrier(device_function)]
        if arrivals == 1:
            result.append(
                statement_from_string(
                    f"tl.atomic_xchg({completion_counter}, {epoch_var}, "
                    "sem='release', scope='gpu')"
                )
            )
        else:
            result.append(
                statement_from_string(
                    f"tl.atomic_add({completion_counter}, 1, "
                    "sem='release', scope='gpu')"
                )
            )
        return result

    root_physical_axis_order = [geometry[0] for geometry in case_geometries]
    root_axis_counts = [domain.axis_counts for domain in root_domains]
    root_events_by_producer: dict[
        int,
        list[
            tuple[
                CountedEventPlan,
                EventContribution,
            ]
        ],
    ] = {}
    producer_events_by_scope: dict[
        int,
        list[
            tuple[
                CountedEventPlan,
                EventContribution,
            ]
        ],
    ] = {}
    for plan in all_counted_event_plans:
        for contributor in plan.contributions:
            if contributor.producer_scope_id is None:
                root_events_by_producer.setdefault(
                    contributor.producer_root, []
                ).append((plan, contributor))
            else:
                producer_events_by_scope.setdefault(
                    contributor.producer_scope_id, []
                ).append((plan, contributor))
    nested_producer_roots = {
        contributor.producer_root
        for contributions in producer_events_by_scope.values()
        for _plan, contributor in contributions
    }
    scheduled_task_roots = {
        root
        for root, traversal in enumerate(root_traversals)
        if any(
            segment.task_relation != traversal
            for segment in cross_loop_schedule.worker_schedule.segments_for_root(root)
        )
    }
    counted_event_uses_by_waiting_root: dict[
        int,
        list[tuple[CountedEventPlan, EventUse]],
    ] = {}
    for plan in counted_event_plans:
        for use_index, use in enumerate(plan.uses):
            if use_index == plan.local_trigger_use:
                continue
            if use.consumer_scope_id is not None:
                continue
            counted_event_uses_by_waiting_root.setdefault(use.consumer_root, []).append(
                (plan, use)
            )
    nested_scope_events_by_consumer: dict[
        int,
        list[tuple[CountedEventPlan, EventUse]],
    ] = {}
    for plan in nested_scope_event_plans:
        for use in plan.uses:
            if use.consumer_scope_id is None:
                continue
            nested_scope_events_by_consumer.setdefault(use.consumer_root, []).append(
                (plan, use)
            )

    def flat_task_coordinates(
        task: str,
        axis_order: tuple[int, ...],
        counts: dict[int, int],
    ) -> dict[int, str]:
        coordinates: dict[int, str] = {}
        multiplier = 1
        for block_id in axis_order:
            count = counts[block_id]
            if count == 1:
                coordinates[block_id] = "0"
            elif multiplier == 1:
                coordinates[block_id] = f"(({task}) % {count})"
            else:
                coordinates[block_id] = f"((({task}) // {multiplier}) % {count})"
            multiplier *= counts[block_id]
        return coordinates

    def flat_task_from_coordinates(
        coordinates: dict[int, str],
        axis_order: tuple[int, ...],
        counts: dict[int, int],
    ) -> str:
        """Flatten logical coordinates in one declared axis order."""
        terms: list[str] = []
        multiplier = 1
        for axis in axis_order:
            count = counts[axis]
            if count != 1:
                coordinate = coordinates[axis]
                terms.append(
                    f"({coordinate})"
                    if multiplier == 1
                    else f"({coordinate}) * {multiplier}"
                )
            multiplier *= count
        return " + ".join(terms) or "0"

    def relation_expression(
        expression: sympy.Expr,
        coordinates: dict[int, str],
    ) -> str:
        """Render the restricted logical-relation expression grammar."""
        if isinstance(expression, sympy.Integer):
            return str(int(expression))
        if isinstance(expression, sympy.Symbol):
            axis = next(
                (
                    axis
                    for axis in coordinates
                    if logical_axis_symbol(axis) == expression
                ),
                None,
            )
            if axis is None:
                raise AssertionError(f"unknown logical relation symbol {expression}")
            return f"({coordinates[axis]})"
        if isinstance(expression, sympy.Add):
            return " + ".join(
                f"({relation_expression(cast('sympy.Expr', term), coordinates)})"
                for term in expression.as_ordered_terms()
            )
        if isinstance(expression, sympy.Mul):
            return " * ".join(
                f"({relation_expression(factor, coordinates)})"
                for factor in expression.as_ordered_factors()
            )
        if expression.func in (sympy.floor, sympy.ceiling):
            numerator, denominator = sympy.fraction(sympy.together(expression.args[0]))
            if not isinstance(denominator, sympy.Integer) or denominator <= 0:
                raise AssertionError(
                    "logical floor/ceiling requires a positive static divisor"
                )
            numerator_expr = relation_expression(numerator, coordinates)
            if expression.func == sympy.floor:
                return f"(({numerator_expr}) // {int(denominator)})"
            return f"(-((-({numerator_expr})) // {int(denominator)}))"
        if expression.func in (sympy.Min, sympy.Max):
            function = "tl.minimum" if expression.func == sympy.Min else "tl.maximum"
            rendered = relation_expression(
                cast("sympy.Expr", expression.args[0]), coordinates
            )
            for argument in expression.args[1:]:
                rendered = (
                    f"{function}(({rendered}), "
                    f"({relation_expression(cast('sympy.Expr', argument), coordinates)}))"
                )
            return rendered
        if isinstance(expression, sympy.Mod):
            numerator, denominator = expression.args
            if not isinstance(denominator, sympy.Integer) or denominator <= 0:
                raise AssertionError(
                    "logical modulo requires a positive static divisor"
                )
            return (
                f"(({relation_expression(cast('sympy.Expr', numerator), coordinates)}) % "
                f"{int(denominator)})"
            )
        if isinstance(expression, sympy.Rational):
            if expression.q == 1:
                return str(expression.p)
            return f"({expression.p} / {expression.q})"
        raise AssertionError(f"unsupported logical relation expression {expression!r}")

    def relation_source_membership(
        bounds: tuple[tuple[int, int, int, int], ...],
        coordinates: dict[int, str],
    ) -> str:
        conditions: list[str] = []
        for axis, begin, end, step in bounds:
            coordinate = coordinates[axis]
            conditions.extend(
                (
                    f"({coordinate}) >= {begin}",
                    f"({coordinate}) < {end}",
                )
            )
            if step != 1:
                conditions.append(f"(({coordinate}) - {begin}) % {step} == 0")
        return " and ".join(conditions) or "True"

    def relation_point_coordinates(
        relation: LogicalRelation,
        source_coordinates: dict[int, str],
    ) -> tuple[dict[int, str], str]:
        """Render an at-most-one-valued relation without task tables."""
        canonical = relation.canonical_single_valued()
        if canonical is None or not canonical.pieces:
            raise AssertionError("event relation is not single-valued")
        memberships: list[str] = []
        values_by_axis: dict[int, list[tuple[str, str]]] = {
            axis: [] for axis in canonical.target_domain.axis_order
        }
        for piece in canonical.pieces:
            source_membership = relation_source_membership(
                piece.source_bounds_items,
                source_coordinates,
            )
            target_memberships: list[str] = []
            piece_values: dict[int, str] = {}
            for axis, begin, end, step in piece.target_ranges:
                if (
                    step != 1
                    or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                    != 1
                ):
                    raise AssertionError("event relation target is not one point")
                value = relation_expression(begin, source_coordinates)
                piece_values[axis] = value
                target_memberships.extend(
                    (
                        f"({value}) >= 0",
                        f"({value}) < {canonical.target_domain.axis_counts[axis]}",
                    )
                )
            membership = " and ".join((source_membership, *target_memberships))
            memberships.append(membership)
            for axis, value in piece_values.items():
                values_by_axis[axis].append((membership, value))

        def select(values: list[tuple[str, str]]) -> str:
            expressions = tuple(dict.fromkeys(value for _membership, value in values))
            if len(expressions) == 1:
                return expressions[0]
            result = values[-1][1]
            for membership, value in reversed(values[:-1]):
                result = f"tl.where({membership}, {value}, {result})"
            return result

        return (
            {axis: select(values) for axis, values in values_by_axis.items()},
            (
                "True"
                if canonical.is_total_function()
                else " or ".join(f"({membership})" for membership in memberships)
            ),
        )

    def relation_flat_target(
        relation: LogicalRelation,
        source_coordinates: dict[int, str],
    ) -> tuple[str, str]:
        target_coordinates, membership = relation_point_coordinates(
            relation,
            source_coordinates,
        )
        return (
            flat_task_from_coordinates(
                target_coordinates,
                relation.target_domain.axis_order,
                relation.target_domain.axis_counts,
            ),
            membership,
        )

    def logical_coordinates_for_physical_task(
        root: int,
        physical_task: str,
    ) -> dict[int, str]:
        """Apply the root's existing PID traversal to one physical task."""
        axis_order = root_physical_axis_order[root]
        counts = root_axis_counts[root]
        case = owner.cases[root]
        group_size = _effective_l2_group_size(case, axis_order, counts)
        if group_size is None:
            return flat_task_coordinates(physical_task, axis_order, counts)

        first_axis, second_axis = axis_order[:2]
        first_count = counts[first_axis]
        second_count = counts[second_axis]
        inner_size = first_count * second_count
        group_span = group_size * second_count
        inner_task = (
            f"(({physical_task}) % {inner_size})"
            if len(axis_order) > 2
            else f"({physical_task})"
        )
        group = f"(({inner_task}) // {group_span})"
        first_in_group = f"({group}) * {group_size}"
        actual_group_size = (
            f"tl.minimum({first_count} - ({first_in_group}), {group_size})"
        )
        within_group = f"(({inner_task}) % {group_span})"
        coordinates = {
            first_axis: (
                f"({first_in_group}) + (({within_group}) % ({actual_group_size}))"
            ),
            second_axis: f"({within_group}) // ({actual_group_size})",
        }
        multiplier = inner_size
        for block_id in axis_order[2:]:
            count = counts[block_id]
            coordinates[block_id] = (
                "0"
                if count == 1
                else f"((({physical_task}) // {multiplier}) % {count})"
            )
            multiplier *= counts[block_id]
        return coordinates

    def physical_task_for_logical_coordinates(
        root: int,
        coordinates: dict[int, str],
    ) -> str:
        """Invert the root's PID traversal without changing its body."""
        axis_order = root_physical_axis_order[root]
        counts = root_axis_counts[root]
        case = owner.cases[root]
        group_size = _effective_l2_group_size(case, axis_order, counts)
        if group_size is None:
            return flat_task_from_coordinates(coordinates, axis_order, counts)

        first_axis, second_axis = axis_order[:2]
        first_count = counts[first_axis]
        second_count = counts[second_axis]
        first_coordinate = coordinates[first_axis]
        second_coordinate = coordinates[second_axis]
        group = f"(({first_coordinate}) // {group_size})"
        first_in_group = f"({group}) * {group_size}"
        actual_group_size = (
            f"tl.minimum({first_count} - ({first_in_group}), {group_size})"
        )
        inner_task = (
            f"({group}) * {group_size * second_count} + "
            f"({second_coordinate}) * ({actual_group_size}) + "
            f"({first_coordinate}) - ({first_in_group})"
        )
        terms = [f"({inner_task})"]
        multiplier = first_count * second_count
        for block_id in axis_order[2:]:
            count = counts[block_id]
            if count != 1:
                terms.append(f"({coordinates[block_id]}) * {multiplier}")
            multiplier *= count
        return " + ".join(terms)

    def logical_task_from_coordinates(
        root: int,
        coordinates: dict[int, str],
    ) -> str:
        return flat_task_from_coordinates(
            coordinates,
            root_domains[root].axis_order,
            root_axis_counts[root],
        )

    def body_with_scope_waits(
        plan: CountedEventPlan,
        use: EventUse,
        body: list[ast.stmt],
        consumer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        assert use.consumer_scope_id is not None
        domain = use.keys.source_domain
        nested_axes = nested_logical_axes(root_domains[use.consumer_root], domain)
        if len(nested_axes) != 1:
            raise AssertionError(
                "nested scope lowering currently requires one loop axis"
            )
        (nested_axis,) = nested_axes
        boundaries = tuple(
            sorted(
                {
                    boundary
                    for piece in use.keys.pieces
                    for axis, begin, end, _step in piece.source_bounds_items
                    if axis == nested_axis
                    for boundary in (begin, end)
                    if 0 < boundary < domain.axis_counts[nested_axis]
                }
            )
        )
        stage_offsets = (0, *boundaries)
        stage_waits: list[tuple[ast.stmt, ...]] = []
        for action_offset in stage_offsets:
            scope_coordinates = {
                **consumer_coordinates,
                nested_axis: str(action_offset),
            }
            event_key, membership = relation_flat_target(
                use.keys,
                scope_coordinates,
            )
            if membership == "False":
                raise AssertionError("nested scope stage has no event key")
            stage_waits.append(
                tuple(
                    _wait_for_counter(
                        device_function=device_function,
                        counter=counted_event_counter(plan, event_key),
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({counted_event_expected_arrivals(plan, event_key)}, tl.uint32)"
                        ),
                        prefix="tile_dependency_scope_wait",
                    )
                )
            )
        return _clone_opaque_statements_with_scope_stages(
            body,
            scope_id=use.consumer_scope_id,
            split_iteration_offsets=boundaries,
            stage_waits=tuple(stage_waits),
        )

    def counted_event_counter(plan: CountedEventPlan, key: str) -> str:
        assert counted_event_arg is not None
        return (
            f"{counted_event_arg} + {counted_event_offsets[plan]} + "
            f"({key}) * {counted_event_key_stride}"
        )

    def counted_event_expected_arrivals(
        plan: CountedEventPlan,
        key: str,
    ) -> str:
        uniform = plan.uniform_arrivals()
        if uniform is not None:
            return str(uniform)
        key_coordinates = flat_task_coordinates(
            key,
            plan.key_domain.axis_order,
            plan.key_domain.axis_counts,
        )
        expressions: list[str] = []
        for contributor in plan.contributions:
            cardinality = contributor.arrivals_per_key
            if cardinality is None:
                raise AssertionError("event fan-in is not symbolically known")
            values, _membership = relation_point_coordinates(
                cardinality,
                key_coordinates,
            )
            expressions.append(values[cardinality.target_domain.axis_order[0]])
        return " + ".join(f"({expression})" for expression in expressions)

    def emit_counted_event_for_key(
        plan: CountedEventPlan,
        key: str,
    ) -> list[ast.stmt]:
        local_use = plan.local_use
        if local_use is None:
            counter = counted_event_counter(plan, key)
            if plan.uniform_arrivals() == 1:
                return [
                    statement_from_string(
                        f"tl.atomic_xchg({counter}, {epoch_var}, "
                        "sem='release', scope='gpu')"
                    )
                ]
            return [
                statement_from_string(
                    f"tl.atomic_add({counter}, 1, sem='release', scope='gpu')"
                )
            ]

        inverse_use = local_use.keys.inverse()
        if inverse_use is None or not inverse_use.is_total_function():
            raise AssertionError("local event use must bijectively cover its consumer")
        if inverse_use.is_positional_bijection():
            consumer_task_expression = key
        else:
            key_coordinates = flat_task_coordinates(
                key,
                plan.key_domain.axis_order,
                plan.key_domain.axis_counts,
            )
            consumer_coordinates, _membership = relation_point_coordinates(
                inverse_use,
                key_coordinates,
            )
            consumer_task_expression = logical_task_from_coordinates(
                local_use.consumer_root,
                consumer_coordinates,
            )
        consumer_task = device_function.new_var(
            "tile_dependency_continuation_task", dce=True
        )
        assignments = [
            statement_from_string(f"{consumer_task} = {consumer_task_expression}")
        ]
        on_ready_root = local_use.consumer_root
        consumer_coordinates = flat_task_coordinates(
            consumer_task,
            root_domains[on_ready_root].axis_order,
            root_axis_counts[on_ready_root],
        )
        consumer_logical_pid = (
            f"{case_offsets[on_ready_root]} + "
            f"{physical_task_for_logical_coordinates(on_ready_root, consumer_coordinates)}"
        )
        consumer_extra_arguments = (consumer_task,)
        previous = device_function.new_var(
            "tile_dependency_continuation_previous", dce=False
        )
        consumer_call = _outline_opaque_tile_body(
            owner,
            device_function,
            root=on_ready_root,
            logical_pid=consumer_logical_pid,
            body=body_with_scope_publications(
                on_ready_root,
                case_bodies[on_ready_root],
                consumer_coordinates,
            ),
            extra_argument_names=consumer_extra_arguments,
        )
        consumer_publications: list[ast.stmt] = []
        for nested_event, nested_contributor in root_events_by_producer.get(
            on_ready_root, ()
        ):
            consumer_publications.extend(
                emit_counted_event_from_producer_coordinates(
                    nested_event,
                    nested_contributor,
                    consumer_coordinates,
                )
            )

        last_arrival_body = [consumer_call]
        if consumer_publications:
            last_arrival_body.append(_publication_barrier(device_function))
            last_arrival_body.extend(consumer_publications)
        last_arrival_body.extend(root_completion_publication(on_ready_root))
        expected_arrivals = plan.uniform_arrivals()
        if expected_arrivals is None:
            raise AssertionError("local execution requires uniform event fan-in")
        if expected_arrivals == 1:
            return [*assignments, *last_arrival_body]
        arrival_counter = counted_event_counter(plan, key)
        return [
            *assignments,
            *_emit_counted_event_on_ready(
                counter=arrival_counter,
                epoch=epoch_var,
                expected_arrivals=expected_arrivals,
                previous=previous,
                on_ready=last_arrival_body,
            ),
        ]

    def emit_counted_event_from_producer_coordinates(
        plan: CountedEventPlan,
        contributor: EventContribution,
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        publication = contributor.producer_to_keys
        if publication is None:
            raise AssertionError("event publication relation is unavailable")
        key, membership = relation_flat_target(
            publication,
            producer_coordinates,
        )
        publications = emit_counted_event_for_key(plan, key)
        if membership == "True":
            return publications
        return [
            create(
                ast.If,
                test=expr_from_string(membership),
                body=publications,
                orelse=[],
            )
        ]

    def body_with_scope_publications(
        root: int,
        body: list[ast.stmt],
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        """Publish nested scope events without moving the owning strand."""
        scope_ids = {
            scope_id
            for scope_id, contributions in producer_events_by_scope.items()
            if any(
                contributor.producer_root == root
                for _plan, contributor in contributions
            )
        }
        if not scope_ids:
            return body
        emitted_scope_ids: set[int] = set()

        def rewrite(loop: ast.For) -> list[ast.stmt] | None:
            scope_id = tile_dependency_scope_id(loop)
            if scope_id is None or scope_id not in scope_ids:
                return None
            if scope_id in emitted_scope_ids:
                raise AssertionError(
                    "one dependency scope must identify one lowered loop"
                )
            contributions = producer_events_by_scope[scope_id]
            producer_scope_domains = {
                contributor.predecessors.target_domain
                for _plan, contributor in contributions
                if contributor.producer_root == root
            }
            if len(producer_scope_domains) != 1:
                raise AssertionError(
                    "nested scope publications must share one producer domain"
                )
            (scope_domain,) = producer_scope_domains
            nested_axes = nested_logical_axes(root_domains[root], scope_domain)
            if (
                len(nested_axes) != 1
                or not isinstance(loop.target, ast.Name)
                or not isinstance(loop.iter, ast.Call)
                or len(loop.iter.args) < 2
            ):
                raise AssertionError(
                    "nested scope publication requires one range-like loop axis"
                )
            nested_axis = nested_axes[0]
            begin = ast.unparse(loop.iter.args[0])
            step = ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
            scope_coordinates = {
                **producer_coordinates,
                nested_axis: f"(({loop.target.id}) - ({begin})) // ({step})",
            }

            publications: list[ast.stmt] = []
            for plan, contributor in producer_events_by_scope[scope_id]:
                publication = contributor.producer_to_keys
                if publication is None:
                    raise AssertionError(
                        "nested event publication relation is unavailable"
                    )
                key, membership = relation_flat_target(
                    publication,
                    scope_coordinates,
                )
                event_publications = emit_counted_event_for_key(plan, key)
                if membership == "True":
                    publications.extend(event_publications)
                else:
                    publications.append(
                        create(
                            ast.If,
                            test=expr_from_string(membership),
                            body=event_publications,
                            orelse=[],
                        )
                    )

            cloned = cast("ast.For", _clone_ast_value(loop))
            cloned.body.extend(
                [
                    _publication_barrier(device_function),
                    *publications,
                ]
            )
            emitted_scope_ids.add(scope_id)
            return [cloned]

        result = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
        if emitted_scope_ids != scope_ids:
            missing = sorted(scope_ids - emitted_scope_ids)
            raise AssertionError(f"missing nested producer scopes {missing}")
        return result

    def scheduled_logical_task_expression(
        root: int,
        schedule_ordinal: str,
    ) -> str | None:
        """Map one root-local schedule position to its logical task ID."""
        if root not in scheduled_task_roots:
            return None
        worker_schedule = cross_loop_schedule.worker_schedule
        root_domain = root_domains[root]
        assignment = worker_schedule.dense_assignment(root)
        if (
            assignment is None
            or assignment[0] != 0
            or assignment[1] != worker_schedule.worker_count
            or assignment[3] != root_domain.size
        ):
            raise AssertionError(
                f"root {root} does not occupy one contiguous schedule interval"
            )
        schedule_begin = assignment[2]
        segments = sorted(
            worker_schedule.segments_for_root(root),
            key=lambda segment: segment.schedule_begin,
        )

        expression = ""
        for segment in reversed(segments):
            ordinal_begin = segment.schedule_begin - schedule_begin
            ordinal_delta = f"(({schedule_ordinal}) - {ordinal_begin})"
            task_offset = ordinal_delta
            membership = (
                f"({ordinal_delta}) >= 0 and ({task_offset}) < {segment.task_count}"
            )
            ordinal_coordinates = flat_task_coordinates(
                task_offset,
                segment.task_relation.source_domain.axis_order,
                segment.task_relation.source_domain.axis_counts,
            )
            task_coordinates, relation_membership = relation_point_coordinates(
                segment.task_relation,
                ordinal_coordinates,
            )
            if relation_membership != "True":
                membership = f"({membership}) and ({relation_membership})"
            segment_task = logical_task_from_coordinates(
                root,
                task_coordinates,
            )
            if not expression:
                expression = segment_task
                continue
            expression = f"tl.where({membership}, {segment_task}, {expression})"
        if not expression:
            raise AssertionError(f"root {root} has no static schedule")
        return expression

    def task_scheduled_body(
        root: int,
        local_task: str,
        logical_pid: str,
        extra_argument_names: tuple[str, ...],
        *,
        force_noinline: bool = False,
    ) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        has_task_scheduling = root in nested_scope_events_by_consumer
        producer_events = tuple(root_events_by_producer.get(root, ()))
        scheduled_local_task = local_task
        scheduled_logical_pid = logical_pid
        scheduled_coordinates: dict[int, str] | None = None
        if producer_events or root in nested_producer_roots:
            has_task_scheduling = True
        if (
            logical_task_expr := scheduled_logical_task_expression(
                root,
                local_task,
            )
        ) is not None:
            scheduled_task = device_function.new_var(
                "tile_dependency_scheduled_logical_task", dce=True
            )
            physical_task = device_function.new_var(
                "tile_dependency_scheduled_physical_task", dce=True
            )
            scheduled_coordinates = flat_task_coordinates(
                scheduled_task,
                root_domains[root].axis_order,
                root_axis_counts[root],
            )
            physical_task_expr = physical_task_for_logical_coordinates(
                root, scheduled_coordinates
            )
            body.extend(
                [
                    statement_from_string(f"{scheduled_task} = {logical_task_expr}"),
                    statement_from_string(f"{physical_task} = {physical_task_expr}"),
                ]
            )
            scheduled_local_task = physical_task
            scheduled_logical_pid = f"{case_offsets[root]} + {physical_task}"
        if scheduled_coordinates is None:
            scheduled_coordinates = logical_coordinates_for_physical_task(
                root, scheduled_local_task
            )
        for (
            incoming_counted_event,
            incoming_use,
        ) in counted_event_uses_by_waiting_root.get(root, ()):
            has_task_scheduling = True
            assert counted_event_arg is not None
            event_key, membership = relation_flat_target(
                incoming_use.keys,
                scheduled_coordinates,
            )
            wait = _wait_for_counter(
                device_function=device_function,
                counter=counted_event_counter(
                    incoming_counted_event,
                    event_key,
                ),
                target=(
                    f"tl.cast({epoch_var}, tl.uint32) * "
                    f"tl.cast({counted_event_expected_arrivals(incoming_counted_event, event_key)}, tl.uint32)"
                ),
                prefix="tile_dependency_keyed_event_wait",
            )
            if not incoming_use.keys.is_total_function():
                body.append(
                    create(
                        ast.If,
                        test=expr_from_string(membership),
                        body=wait,
                        orelse=[],
                    )
                )
            else:
                body.extend(wait)
        nested_event_uses = nested_scope_events_by_consumer.get(root, ())
        if nested_event_uses:
            # Instrument the original logical loop before segmentation.
            # Split ranges then retain the original scope-coordinate
            # expression instead of rebasing publication IDs per segment.
            scheduled_root_body = body_with_scope_publications(
                root,
                case_bodies[root],
                scheduled_coordinates,
            )
            for scope_plan, scope_use in sorted(
                nested_event_uses,
                key=lambda item: (
                    item[1].consumer_scope_id
                    if item[1].consumer_scope_id is not None
                    else -1
                ),
            ):
                scheduled_root_body = body_with_scope_waits(
                    scope_plan,
                    scope_use,
                    scheduled_root_body,
                    scheduled_coordinates,
                )
            opaque_call = _outline_cross_loop_region(
                device_function,
                name_hint=f"tile_dependency_root_{root}",
                body=[
                    statement_from_string(
                        f"{owner.shared_pid_var} = {scheduled_logical_pid}"
                    ),
                    *scheduled_root_body,
                ],
                extra_argument_names=extra_argument_names,
            )
        else:
            opaque_call = _outline_opaque_tile_body(
                owner,
                device_function,
                root=root,
                logical_pid=scheduled_logical_pid,
                body=body_with_scope_publications(
                    root,
                    case_bodies[root],
                    scheduled_coordinates,
                ),
                extra_argument_names=extra_argument_names,
                noinline=force_noinline,
            )
        body.append(opaque_call)
        if producer_events:
            has_task_scheduling = True
            body.append(_publication_barrier(device_function))
        for counted_event, contributor in producer_events:
            body.extend(
                emit_counted_event_from_producer_coordinates(
                    counted_event,
                    contributor,
                    scheduled_coordinates,
                )
            )
        if not has_task_scheduling:
            return body
        return [
            _outline_cross_loop_region(
                device_function,
                name_hint=f"tile_dependency_root_{root}_scheduled_task",
                body=body,
                extra_argument_names=extra_argument_names,
                noinline=True,
            )
        ]

    dense_assignment_by_root: dict[int, tuple[int, int, int]] = {}
    for root, root_domain in enumerate(root_domains):
        assignment = cross_loop_schedule.worker_schedule.dense_assignment(root)
        if assignment is None:
            if cross_loop_schedule.worker_schedule.segments_for_root(root):
                raise exc.InvalidConfig(
                    f"{CROSS_LOOP_SCHEDULE_CONFIG}="
                    f"{CROSS_LOOP_SCHEDULE_STATIC_PIPELINE!r} cannot lower "
                    f"root {root}'s non-dense worker assignment"
                )
            continue
        worker_begin, worker_count, schedule_begin, task_count = assignment
        if task_count != root_domain.size or schedule_begin % worker_count:
            raise exc.InvalidConfig(
                f"{CROSS_LOOP_SCHEDULE_CONFIG}="
                f"{CROSS_LOOP_SCHEDULE_STATIC_PIPELINE!r} does not "
                f"support root {root}'s worker assignment"
            )
        dense_assignment_by_root[root] = (
            worker_begin,
            worker_count,
            root_domain.size,
        )

    def static_root_body(root: int) -> list[ast.stmt]:
        if root in consumed_on_ready_roots:
            return []
        assignment = dense_assignment_by_root.get(root)
        if assignment is None:
            return []
        worker_begin, worker_count, task_count = assignment
        task_dispatch: list[ast.stmt]
        if task_count == 1:
            task_dispatch = task_scheduled_body(
                root,
                "0",
                str(case_offsets[root]),
                (epoch_var,),
                force_noinline=True,
            )
        else:
            local_task = f"({strategy.virtual_pid_var}) - {case_offsets[root]}"
            task_dispatch = [
                create(
                    ast.For,
                    target=create(
                        ast.Name,
                        id=strategy.virtual_pid_var,
                        ctx=ast.Store(),
                    ),
                    iter=expr_from_string(
                        f"tl.range((({worker}) - {worker_begin}) + "
                        f"({case_offsets[root]}), "
                        f"({case_offsets[root] + task_count}), {worker_count})"
                    ),
                    body=task_scheduled_body(
                        root,
                        local_task,
                        strategy.virtual_pid_var,
                        (strategy.virtual_pid_var,),
                    ),
                    orelse=[],
                    type_comment=None,
                )
            ]
        incoming_roots = root_completion_incoming.get(root, ())
        publishes_completion = root in root_completion_indices
        if (
            worker_begin == 0
            and worker_count == launch_worker_count
            and not incoming_roots
            and not publishes_completion
        ):
            return task_dispatch

        active_body = _wait_for_dependencies(
            device_function=device_function,
            dependencies=root_completion_input_dependencies(root),
            prefix="tile_dependency_root_completion_wait",
        )
        active_body.extend(task_dispatch)
        if publishes_completion:
            active_body.extend(root_completion_publication(root))
        condition = (
            f"({worker}) == {worker_begin}"
            if worker_count == 1
            else (
                f"({worker}) >= {worker_begin} and "
                f"({worker}) < {worker_begin + worker_count}"
            )
        )
        return [
            create(
                ast.If,
                test=expr_from_string(condition),
                body=active_body,
                orelse=[],
            )
        ]

    for root_begin, root_end in stage_root_ranges:
        for root in range(root_begin, root_end):
            result.extend(static_root_body(root))
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
