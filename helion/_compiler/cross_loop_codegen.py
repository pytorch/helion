from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.utils._sympy.functions import CeilDiv

from .. import exc
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cross_loop_scheduler import _SOURCE_LAUNCH_STAGE
from .cross_loop_scheduler import ReadinessConsumer
from .cross_loop_scheduler import ReadinessCounterPlan
from .cross_loop_scheduler import ReadinessProducer
from .cross_loop_scheduler import WorkerInterval
from .cross_loop_scheduler import WorkerScheduleSegment
from .cross_loop_scheduler import _normalize_intervals
from .cross_loop_scheduler import _parametric_event_frontier_root_order
from .cross_loop_scheduler import _parametric_event_frontier_schedule_geometry
from .cross_loop_scheduler import _parametric_root_major_schedule_geometry
from .cross_loop_scheduler import _root_schedule_traversal
from .cross_loop_scheduler import _supports_parameterized_fan_in_one_counter
from .cross_loop_scheduler import _transient_source_schedule_segment
from .cross_loop_scheduler import build_static_pipeline_plan
from .cross_loop_scheduler import root_barrier_publication_plan
from .device_function import TensorArg
from .host_function import HostFunction
from .program_id import _clone_ast_value
from .program_id import _clone_stmt
from .program_id import typed_program_id
from .tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_coordinate_domains
from .tile_dependency import nested_logical_axes
from .tile_dependency import pid_task_order
from .tile_dependency import tile_dependency_site_id
from .tile_strategy import L2GroupingProgramIDs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from .device_function import DeviceFunction
    from .program_id import ForEachProgramID
    from .program_id import PersistentProgramIDs
    from .program_id import PIDInfo
    from .program_id import ProgramIDs


# Independent readiness counters occupy distinct cache lines so polling one
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
                site_id := getattr(value, TILE_DEPENDENCY_SITE_ID_ATTR, None)
            ) is not None:
                setattr(cloned, TILE_DEPENDENCY_SITE_ID_ATTR, site_id)
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


def _clone_opaque_statements_with_loop_segments(
    body: list[ast.stmt],
    *,
    site_id: int,
    split_iteration_offsets: tuple[int, ...],
    segment_waits: tuple[tuple[ast.stmt, ...], ...],
) -> list[ast.stmt]:
    """Split one stable DeviceIR execution-site loop and wait before each segment."""
    if len(segment_waits) != len(split_iteration_offsets) + 1:
        raise AssertionError("each loop segment requires one wait")
    scheduled = False

    def rewrite(loop: ast.For) -> list[ast.stmt] | None:
        nonlocal scheduled
        if tile_dependency_site_id(loop) != site_id:
            return None
        if scheduled:
            raise AssertionError("one dependency site must identify one lowered loop")
        if not isinstance(loop.iter, ast.Call) or len(loop.iter.args) < 2:
            raise AssertionError("nested loop scheduling requires a range-like loop")
        begin = ast.unparse(loop.iter.args[0])
        step = ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
        split_offsets = tuple(
            f"({begin}) + ({offset}) * ({step})" for offset in split_iteration_offsets
        )
        boundaries = (None, *split_offsets, None)
        result: list[ast.stmt] = []
        for index, waits in enumerate(segment_waits):
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
        present_site_ids = sorted(
            found_site_id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.For)
            if (found_site_id := tile_dependency_site_id(node)) is not None
        )
        raise AssertionError(
            f"missing dependency site {site_id}; found {present_site_ids}"
        )
    return cloned


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
) -> list[tuple[int | sympy.Expr, int]] | None:
    env = CompileEnvironment.current()
    task_families = HostFunction.current().device_ir.task_families
    if root >= len(task_families):
        return None
    task_family = task_families[root]
    result: list[tuple[int | sympy.Expr, int]] = []
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
            numel = numel_expr._sympy_()
        elif getattr(numel_expr, "is_number", False):
            numel = int(numel_expr)
        elif isinstance(numel_expr, sympy.Expr):
            numel = numel_expr
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
) -> (
    tuple[
        tuple[int, ...],
        dict[int, int | sympy.Expr],
        dict[int, int],
    ]
    | None
):
    axes = _static_case_axes(owner, root, device_function)
    if axes is None:
        return None
    infos = _case_pid_info(owner.cases[root])
    axis_order = tuple(info.block_id for info in infos)
    axis_counts: dict[int, int | sympy.Expr] = {
        info.block_id: (
            (numel + block - 1) // block
            if isinstance(numel, int)
            else cast("sympy.Expr", CeilDiv(numel, block))
        )
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
    axis_counts: dict[int, int | sympy.Expr],
) -> int | None:
    """Return the nontrivial L2 grouping applied by one root's PID task order."""
    if not isinstance(case, L2GroupingProgramIDs) or len(axis_order) < 2:
        return None
    first_axis, second_axis = axis_order[:2]
    first_count = axis_counts[first_axis]
    second_count = axis_counts[second_axis]
    if not isinstance(first_count, int) or not isinstance(second_count, int):
        return None
    if second_count == 1 or case.group_size >= first_count:
        return None
    return case.group_size


def _root_task_orders(
    owner: ForEachProgramID,
    root_domains: tuple[CoordinateDomain, ...],
    case_geometries: tuple[
        tuple[tuple[int, ...], dict[int, int | sympy.Expr], dict[int, int]], ...
    ],
) -> tuple[CoordinateRelation, ...] | None:
    """Bind each logical root domain to its configured PID task order."""
    if len(root_domains) != len(case_geometries):
        return None
    result: list[CoordinateRelation] = []
    for root, (domain, geometry) in enumerate(
        zip(root_domains, case_geometries, strict=True)
    ):
        pid_axis_order, axis_counts, block_sizes = geometry
        if domain.parameter_symbols and (
            len(pid_axis_order) != 1
            or isinstance(owner.cases[root], L2GroupingProgramIDs)
        ):
            return None
        if (
            set(domain.axis_order) != set(pid_axis_order)
            or domain.axis_count_expressions
            != {axis: axis_counts[axis] for axis in domain.axis_order}
            or domain.block_sizes
            != {axis: block_sizes[axis] for axis in domain.axis_order}
        ):
            return None
        case = owner.cases[root]
        l2_group_size = _effective_l2_group_size(
            case,
            pid_axis_order,
            axis_counts,
        )
        result.append(
            pid_task_order(
                domain,
                pid_axis_order,
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


def _emit_final_arrival_continuation(
    *,
    counter: str,
    epoch: str,
    expected_arrivals: int,
    previous: str,
    continuation_body: list[ast.stmt],
) -> list[ast.stmt]:
    """Publish one arrival and run the continuation on the final arrival."""
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
            body=continuation_body,
            orelse=[],
        ),
    ]


def _publication_sync(device_function: DeviceFunction) -> ast.stmt:
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


def _triton_root_requires_kernel_scope(
    body: list[ast.stmt],
    target_device_capability: tuple[int, int] | None,
) -> bool:
    """Return whether a Triton root may allocate kernel-scoped resources.

    Blackwell lowers sufficiently large ``tl.dot`` operations through tensor
    memory.  Triton's tensor-memory allocation must remain in the kernel body;
    placing the operation in an outlined device function fails during lowering.
    Conservatively keep every Blackwell dot root in kernel scope because the
    eventual tensor-memory choice is made after Helion emits its AST.
    """
    if target_device_capability is None or target_device_capability[0] < 10:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr in {"dot", "dot_scaled"}
        for statement in body
        for node in ast.walk(statement)
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
    schedule = device_function.config.cross_loop_schedule
    if schedule == "barrier":
        device_function.has_barrier = True
        return owner._emit_phase_loops(strategy, device_function, total_expr)
    if schedule != "static_pipeline":
        raise exc.InvalidConfig(f"unknown cross_loop_schedule value {schedule!r}")

    configured_case_geometries = tuple(
        _static_case_geometry(owner, root, device_function)
        for root in range(len(owner.cases))
    )
    if any(geometry is None for geometry in configured_case_geometries):
        raise exc.InvalidConfig(
            "cross_loop_schedule='static_pipeline' requires representable "
            "top-level task counts"
        )
    case_geometries = tuple(
        geometry for geometry in configured_case_geometries if geometry is not None
    )
    program = typed_program_id(0)
    worker = program
    epoch_var = device_function.new_var("tile_dependency_epoch", dce=False)
    base_body = cast(
        "list[ast.stmt]",
        owner._prepare_persistent_body(
            device_function.body,
            device_function,
            strategy.virtual_pid_var,
        ),
    )
    case_bodies = _extract_case_bodies(owner, base_body)
    opaque_case_fingerprints = tuple(_ast_fingerprint(body) for body in case_bodies)
    target_device_capability = (
        CompileEnvironment.current().config_spec.target_device_capability
    )
    kernel_scope_roots = frozenset(
        root
        for root, body in enumerate(case_bodies)
        if _triton_root_requires_kernel_scope(body, target_device_capability)
    )
    dependency_graph = HostFunction.current().device_ir.tile_dependency_graph
    assert dependency_graph is not None
    indexing = device_function.config.get("indexing", ())

    def uses_tensor_descriptor(memory_op_index: int) -> bool:
        if isinstance(indexing, str):
            return indexing == "tensor_descriptor"
        return (
            isinstance(indexing, (list, tuple))
            and 0 <= memory_op_index < len(indexing)
            and indexing[memory_op_index] == "tensor_descriptor"
        )

    unpublishable_site_ids = frozenset(
        site_id
        for access in dependency_graph.accesses
        if access.kind == "store" and uses_tensor_descriptor(access.memory_op_index)
        for site_id in dependency_graph.site_ids_by_access[access.access_id]
    )
    publishable_site_ids = (
        frozenset(
            site.site_id
            for site in dependency_graph.execution_sites
            if not site.is_root and site.site_id not in unpublishable_site_ids
        )
        if unpublishable_site_ids
        else None
    )
    configured_worker_count = CompileEnvironment.current().config_spec.num_sm * cast(
        "int", device_function.config.get("num_sm_multiplier", 1)
    )
    root_axis_geometry: dict[int, tuple[int | sympy.Expr, int]] = {}
    for _axis_order, axis_counts, block_sizes in case_geometries:
        for block_id, task_count in axis_counts.items():
            geometry = (task_count, block_sizes[block_id])
            previous = root_axis_geometry.setdefault(block_id, geometry)
            if previous != geometry:
                raise AssertionError(
                    f"inconsistent configured geometry for block axis {block_id}"
                )
    axis_geometry: dict[int, tuple[int | sympy.Expr, int]] = {}
    for block_id in range(len(CompileEnvironment.current().block_sizes)):
        geometry = root_axis_geometry.get(block_id)
        if geometry is None:
            geometry = _static_block_axis_geometry(block_id, device_function)
        if geometry is not None:
            axis_geometry[block_id] = geometry
    configured_root_domains, site_domains = instantiate_coordinate_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if any(domain is None for domain in configured_root_domains):
        raise exc.InvalidConfig(
            "cross_loop_schedule='static_pipeline' requires representable root domains"
        )
    root_domains = tuple(
        domain for domain in configured_root_domains if domain is not None
    )
    parameterized_root_domains = any(
        domain.parameter_symbols for domain in root_domains
    )
    case_offsets: list[int] = []
    case_offset_expressions: list[sympy.Expr] = []
    running_offset_expression: sympy.Expr = sympy.Integer(0)
    for domain in root_domains:
        case_offset_expressions.append(running_offset_expression)
        running_offset_expression = sympy.simplify(
            sympy.Add(running_offset_expression, domain.size_expr)
        )
    if not parameterized_root_domains:
        case_offsets = [int(offset) for offset in case_offset_expressions]
    case_offset_strings = [
        device_function.sympy_expr(offset) for offset in case_offset_expressions
    ]
    root_task_orders = _root_task_orders(
        owner,
        root_domains,
        case_geometries,
    )
    if root_task_orders is None:
        raise exc.InvalidConfig(
            "cross_loop_schedule='static_pipeline' requires a "
            "representable root PID task order"
        )
    static_pipeline_plan = build_static_pipeline_plan(
        dependency_graph=dependency_graph,
        root_task_orders=root_task_orders,
        site_domains=site_domains,
        worker_count=configured_worker_count,
        publishable_site_ids=publishable_site_ids,
        allow_transient_source=(
            CompileEnvironment.current().backend_name == "triton"
            and CompileEnvironment.current().device.type == "cuda"
            and CompileEnvironment.current().settings.persistent_reserved_sms == 0
            and device_function.config.get("num_sm_multiplier", 1) == 1
        ),
    )
    parameterized_root_major_geometry = _parametric_root_major_schedule_geometry(
        static_pipeline_plan.worker_schedule
    )
    parameterized_event_frontier_geometry = (
        _parametric_event_frontier_schedule_geometry(
            static_pipeline_plan.worker_schedule
        )
    )
    has_parameterized_schedule = (
        parameterized_root_major_geometry is not None
        or parameterized_event_frontier_geometry is not None
    )
    if parameterized_root_domains and (
        parameterized_root_major_geometry is None
        and parameterized_event_frontier_geometry is None
    ):
        raise exc.InvalidConfig(
            "cross_loop_schedule='static_pipeline' cannot lower this "
            "parameterized worker schedule"
        )
    if parameterized_root_domains and (
        static_pipeline_plan.transient_source_root is not None
        or any(
            not _supports_parameterized_fan_in_one_counter(plan)
            for plan in static_pipeline_plan.readiness_counters
        )
    ):
        raise AssertionError("parameterized lowering received an unproved counter plan")
    root_barrier_edges = static_pipeline_plan.root_barrier_edges
    all_readiness_counter_plans = static_pipeline_plan.readiness_counters
    if parameterized_event_frontier_geometry is not None:
        expected_root_order = _parametric_event_frontier_root_order(
            root_task_orders,
            all_readiness_counter_plans,
            root_barrier_edges,
        )
        if expected_root_order != tuple(
            segment.root
            for segment, _phase, _task_count in parameterized_event_frontier_geometry
        ):
            raise AssertionError(
                "parameterized event-frontier schedule violates its prerequisites"
            )
    nested_loop_counter_plans = tuple(
        plan
        for plan in all_readiness_counter_plans
        if any(
            readiness_consumer.consumer_site_id is not None
            for readiness_consumer in plan.consumers
        )
    )
    readiness_counter_plans = tuple(
        plan
        for plan in all_readiness_counter_plans
        if all(
            readiness_consumer.consumer_site_id is None
            for readiness_consumer in plan.consumers
        )
    )
    launch_worker_count = static_pipeline_plan.worker_schedule.worker_count
    transient_source_root = static_pipeline_plan.transient_source_root
    transient_source_segment = (
        None
        if transient_source_root is None
        else _transient_source_schedule_segment(
            static_pipeline_plan.worker_schedule,
            transient_source_root,
        )
    )
    launch_stage_zero_segments = tuple(
        segment
        for segment in static_pipeline_plan.worker_schedule.segments
        if segment.launch_stage == _SOURCE_LAUNCH_STAGE
    )
    if transient_source_root is None and launch_stage_zero_segments:
        raise AssertionError(
            "a launch-stage-zero segment requires a transient source role"
        )
    if transient_source_root is not None and (
        transient_source_segment is None
        or launch_stage_zero_segments != (transient_source_segment,)
    ):
        raise AssertionError(
            "a transient source must have one exact launch-stage-zero segment"
        )
    transient_source_task_count = (
        transient_source_segment.task_count
        if transient_source_segment is not None
        else 0
    )
    launch_program_count = launch_worker_count + transient_source_task_count
    resident_grid_size_expr = strategy.grid_size_expr
    if transient_source_root is not None:
        worker = device_function.new_var("tile_dependency_resident_worker", dce=True)

    root_barrier_producer_roots = sorted(
        {producer for producer, _consumer in root_barrier_edges}
    )
    root_publication_plans = {
        root: root_barrier_publication_plan(
            static_pipeline_plan.worker_schedule,
            root,
        )
        for root in root_barrier_producer_roots
        if root != transient_source_root
    }
    continuation_task_count_by_root: dict[int, int] = {}
    for plan in readiness_counter_plans:
        readiness_consumer = plan.continuation_consumer
        if readiness_consumer is None:
            continue
        if not readiness_consumer.keys_by_consumer.is_total_function():
            raise AssertionError(
                "a final-arrival continuation must cover its complete root"
            )
        continuation_task_count_by_root[readiness_consumer.consumer_root] = (
            readiness_consumer.keys_by_consumer.source_domain.size
        )

    def root_barrier_arrival_count(root: int) -> int:
        scheduled_arrivals = (
            transient_source_task_count
            if root == transient_source_root
            else root_publication_plans[root].resident_arrival_count
        )
        return scheduled_arrivals + continuation_task_count_by_root.get(root, 0)

    # Reject grids that cannot residently fit when the device is otherwise
    # idle. Concurrent-stream residency remains an explicit unresolved
    # contract for this non-cooperative static lowering.
    device_function.triton_minimum_resident_programs = resident_grid_size_expr
    device_function.preamble.extend(strategy._persistent_setup_statements(total_expr))
    if transient_source_root is not None:
        strategy.grid_size_expr = (
            f"({resident_grid_size_expr} + {transient_source_task_count})"
        )
    readiness_counter_offsets: dict[ReadinessCounterPlan, int | sympy.Expr] = {}
    readiness_counter_count: int | sympy.Expr = 0
    readiness_counter_stride = _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    for plan in all_readiness_counter_plans:
        readiness_counter_offsets[plan] = readiness_counter_count
        readiness_counter_count = sympy.simplify(
            sympy.Add(
                sympy.sympify(readiness_counter_count),
                sympy.Mul(
                    plan.readiness_key_count_expr,
                    readiness_counter_stride,
                ),
            )
        )
    root_barrier_indices = {
        root: index for index, root in enumerate(root_barrier_producer_roots)
    }
    state_count: int | sympy.Expr = 0

    def reserve_state(count: int | sympy.Expr) -> int | sympy.Expr | None:
        nonlocal state_count
        if sympy.simplify(count) == 0:
            return None
        if isinstance(state_count, int):
            state_count = (
                (state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
                // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
                * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
            )
        else:
            state_count = sympy.simplify(
                sympy.Mul(
                    CeilDiv(state_count, _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS),
                    _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS,
                )
            )
        offset = state_count
        state_count = sympy.simplify(
            sympy.Add(sympy.sympify(state_count), sympy.sympify(count))
        )
        return offset

    root_barrier_count = (
        len(root_barrier_producer_roots) * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    if not has_parameterized_schedule:
        readiness_counter_state_offset = reserve_state(readiness_counter_count)
        root_barrier_state_offset = reserve_state(root_barrier_count)
    else:
        # Cumulative root barriers need shape-stable addresses across graph
        # replay.  Parameter-sized fan-in-one counters contain absolute epoch
        # values, so they may safely occupy the variable tail.
        root_barrier_state_offset = reserve_state(root_barrier_count)
        readiness_counter_state_offset = reserve_state(readiness_counter_count)
    epoch_state_count = launch_program_count if transient_source_root is None else 0
    static_state_base = str(
        (epoch_state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
        // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    state_arg = _register_cross_loop_state(
        device_function,
        name_hint="tile_dependency_state",
        numel=(
            f"{static_state_base} + "
            f"{HostFunction.current().sympy_expr(sympy.sympify(state_count))}"
        ),
        dtype=_CROSS_LOOP_COUNTER_DTYPE,
    )
    dispatch_ticket_arg = (
        _register_cross_loop_state(
            device_function,
            name_hint="tile_dependency_dispatch_ticket",
            numel="1",
            dtype=torch.uint64,
        )
        if transient_source_root is not None
        else None
    )

    def state_section(offset: int | sympy.Expr | None) -> str | None:
        if offset is None:
            return None
        offset_text = (
            str(offset)
            if isinstance(offset, int)
            else device_function.sympy_expr(offset)
        )
        return f"{state_arg} + ({static_state_base}) + {offset_text}"

    epoch_arg = state_arg
    readiness_counter_arg = state_section(readiness_counter_state_offset)
    root_barrier_counter_arg = state_section(root_barrier_state_offset)

    dispatch_ticket: str | None = None
    if transient_source_root is None:
        result: list[ast.stmt] = [
            statement_from_string(f"{epoch_var} = tl.load({epoch_arg} + {worker}) + 1")
        ]
    else:
        assert dispatch_ticket_arg is not None
        raw_dispatch_ticket = device_function.new_var(
            "tile_dependency_raw_dispatch_ticket", dce=False
        )
        dispatch_ticket = device_function.new_var(
            "tile_dependency_dispatch_ticket", dce=True
        )
        result = [
            statement_from_string(
                f"{raw_dispatch_ticket} = tl.atomic_add("
                f"{dispatch_ticket_arg}, 1, sem='relaxed', scope='gpu')"
            ),
            statement_from_string(
                f"{dispatch_ticket} = tl.cast("
                f"{raw_dispatch_ticket} % tl.cast("
                f"{launch_program_count}, tl.uint64), tl.int32)"
            ),
            statement_from_string(
                f"{epoch_var} = tl.cast("
                f"{raw_dispatch_ticket} // tl.cast("
                f"{launch_program_count}, tl.uint64) + 1, tl.uint32)"
            ),
            statement_from_string(
                f"{worker} = {dispatch_ticket} - {transient_source_task_count}"
            ),
        ]
    continuation_roots = {
        continuation_consumer.consumer_root
        for plan in readiness_counter_plans
        if (continuation_consumer := plan.continuation_consumer) is not None
    }

    root_barrier_incoming: dict[int, tuple[int, ...]] = {
        consumer: tuple(
            sorted(
                producer
                for producer, target in root_barrier_edges
                if target == consumer
            )
        )
        for consumer in {consumer for _producer, consumer in root_barrier_edges}
    }

    def root_barrier_counter(root: int) -> str:
        assert root_barrier_counter_arg is not None
        return (
            f"{root_barrier_counter_arg} + "
            f"{root_barrier_indices[root] * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}"
        )

    def root_barrier_dependency(root: int) -> tuple[str, str]:
        arrivals = root_barrier_arrival_count(root)
        return (
            root_barrier_counter(root),
            f"tl.cast({epoch_var}, tl.uint32) * tl.cast({arrivals}, tl.uint32)",
        )

    def root_barrier_input_dependencies(
        root: int,
    ) -> tuple[tuple[str, str], ...]:
        producers = root_barrier_incoming.get(root, ())
        return tuple(root_barrier_dependency(producer) for producer in producers)

    def root_barrier_publication(root: int) -> list[ast.stmt]:
        if root not in root_barrier_indices:
            return []
        barrier_counter = root_barrier_counter(root)
        arrivals = root_barrier_arrival_count(root)
        result = [_publication_sync(device_function)]
        if arrivals == 1:
            result.append(
                statement_from_string(
                    f"tl.atomic_xchg({barrier_counter}, {epoch_var}, "
                    "sem='release', scope='gpu')"
                )
            )
        else:
            result.append(
                statement_from_string(
                    f"tl.atomic_add({barrier_counter}, 1, sem='release', scope='gpu')"
                )
            )
        return result

    root_pid_axis_orders = [geometry[0] for geometry in case_geometries]
    root_axis_counts = [domain.axis_count_expressions for domain in root_domains]
    root_counters_by_producer: dict[
        int,
        list[
            tuple[
                ReadinessCounterPlan,
                ReadinessProducer,
            ]
        ],
    ] = {}
    producer_counters_by_site: dict[
        int,
        list[
            tuple[
                ReadinessCounterPlan,
                ReadinessProducer,
            ]
        ],
    ] = {}
    for plan in all_readiness_counter_plans:
        for readiness_producer in plan.producers:
            if readiness_producer.producer_site_id is None:
                root_counters_by_producer.setdefault(
                    readiness_producer.producer_root, []
                ).append((plan, readiness_producer))
            else:
                producer_counters_by_site.setdefault(
                    readiness_producer.producer_site_id, []
                ).append((plan, readiness_producer))
    nested_producer_roots = {
        readiness_producer.producer_root
        for readiness_producers in producer_counters_by_site.values()
        for _plan, readiness_producer in readiness_producers
    }
    root_schedule_traversals = {}
    scheduled_task_roots: set[int] = set()
    if not has_parameterized_schedule:
        for root, task_order in enumerate(root_task_orders):
            segments = static_pipeline_plan.worker_schedule.segments_for_root(root)
            if not segments:
                continue
            traversal = _root_schedule_traversal(segments, task_order)
            if traversal is None:
                raise exc.InvalidConfig(
                    "cross_loop_schedule='static_pipeline' does not have a "
                    f"symbolically bijective traversal for root {root}"
                )
            root_schedule_traversals[root] = traversal
            if not traversal.matches_reference:
                scheduled_task_roots.add(root)
    readiness_consumers_by_root: dict[
        int,
        list[tuple[ReadinessCounterPlan, ReadinessConsumer]],
    ] = {}
    for plan in readiness_counter_plans:
        for consumer_index, readiness_consumer in enumerate(plan.consumers):
            if consumer_index == plan.continuation_consumer_index:
                continue
            if readiness_consumer.consumer_site_id is not None:
                continue
            readiness_consumers_by_root.setdefault(
                readiness_consumer.consumer_root, []
            ).append((plan, readiness_consumer))
    nested_loop_counters_by_consumer: dict[
        int,
        list[tuple[ReadinessCounterPlan, ReadinessConsumer]],
    ] = {}
    for plan in nested_loop_counter_plans:
        for readiness_consumer in plan.consumers:
            if readiness_consumer.consumer_site_id is None:
                continue
            nested_loop_counters_by_consumer.setdefault(
                readiness_consumer.consumer_root, []
            ).append((plan, readiness_consumer))

    def flat_task_coordinates(
        task: str,
        axis_order: tuple[int, ...],
        counts: Mapping[int, int | sympy.Expr],
    ) -> dict[int, str]:
        coordinates: dict[int, str] = {}
        multiplier: int | sympy.Expr = 1
        for block_id in axis_order:
            count = counts[block_id]
            count_text = (
                str(count)
                if isinstance(count, int)
                else device_function.sympy_expr(count)
            )
            multiplier_text = (
                str(multiplier)
                if isinstance(multiplier, int)
                else device_function.sympy_expr(multiplier)
            )
            if count == 1:
                coordinates[block_id] = "0"
            elif multiplier == 1:
                coordinates[block_id] = f"(({task}) % {count_text})"
            else:
                coordinates[block_id] = (
                    f"((({task}) // {multiplier_text}) % {count_text})"
                )
            multiplier = cast(
                "sympy.Expr",
                sympy.Mul(sympy.sympify(multiplier), sympy.sympify(count)),
            )
        return coordinates

    def flat_task_from_coordinates(
        coordinates: dict[int, str],
        axis_order: tuple[int, ...],
        counts: Mapping[int, int | sympy.Expr],
    ) -> str:
        """Flatten logical coordinates in one declared axis order."""
        terms: list[str] = []
        multiplier: int | sympy.Expr = 1
        for axis in axis_order:
            count = counts[axis]
            if count != 1:
                coordinate = coordinates[axis]
                multiplier_text = (
                    str(multiplier)
                    if isinstance(multiplier, int)
                    else device_function.sympy_expr(multiplier)
                )
                terms.append(
                    f"({coordinate})"
                    if multiplier == 1
                    else f"({coordinate}) * {multiplier_text}"
                )
            multiplier = cast(
                "sympy.Expr",
                sympy.Mul(sympy.sympify(multiplier), sympy.sympify(count)),
            )
        return " + ".join(terms) or "0"

    def relation_expression(
        expression: sympy.Expr,
        coordinates: dict[int, str],
    ) -> str:
        """Render the restricted coordinate-relation expression grammar."""
        if isinstance(expression, sympy.Integer):
            return str(int(expression))
        coordinate_symbols = frozenset(
            coordinate_axis_symbol(axis) for axis in coordinates
        )
        if expression.free_symbols and expression.free_symbols.isdisjoint(
            coordinate_symbols
        ):
            return f"({device_function.sympy_expr(expression)})"
        if isinstance(expression, sympy.Symbol):
            axis = next(
                (
                    axis
                    for axis in coordinates
                    if coordinate_axis_symbol(axis) == expression
                ),
                None,
            )
            if axis is not None:
                return f"({coordinates[axis]})"
            return f"({device_function.sympy_expr(expression)})"
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
        raise AssertionError(
            f"unsupported coordinate-relation expression {expression!r}"
        )

    def relation_source_membership(
        bounds: tuple[tuple[int, int | sympy.Expr, int | sympy.Expr, int], ...],
        coordinates: dict[int, str],
    ) -> str:
        conditions: list[str] = []
        for axis, begin, end, step in bounds:
            coordinate = coordinates[axis]
            begin_text = relation_expression(sympy.sympify(begin), coordinates)
            end_text = relation_expression(sympy.sympify(end), coordinates)
            conditions.extend(
                (
                    f"({coordinate}) >= {begin_text}",
                    f"({coordinate}) < {end_text}",
                )
            )
            if step != 1:
                conditions.append(f"(({coordinate}) - {begin_text}) % {step} == 0")
        return " and ".join(conditions) or "True"

    def relation_point_coordinates(
        relation: CoordinateRelation,
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
                target_count = relation_expression(
                    sympy.sympify(canonical.target_domain.axis_count_expressions[axis]),
                    source_coordinates,
                )
                target_memberships.extend(
                    (
                        f"({value}) >= 0",
                        f"({value}) < {target_count}",
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
        relation: CoordinateRelation,
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
                relation.target_domain.axis_count_expressions,
            ),
            membership,
        )

    def logical_coordinates_for_pid_task(
        root: int,
        pid_task: str,
    ) -> dict[int, str]:
        """Apply the root's configured PID task order to one PID task."""
        axis_order = root_pid_axis_orders[root]
        counts = root_axis_counts[root]
        case = owner.cases[root]
        group_size = _effective_l2_group_size(case, axis_order, counts)
        if group_size is None:
            return flat_task_coordinates(pid_task, axis_order, counts)

        first_axis, second_axis = axis_order[:2]
        first_count = counts[first_axis]
        second_count = counts[second_axis]
        inner_size = first_count * second_count
        group_span = group_size * second_count
        inner_task = (
            f"(({pid_task}) % {inner_size})" if len(axis_order) > 2 else f"({pid_task})"
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
                "0" if count == 1 else f"((({pid_task}) // {multiplier}) % {count})"
            )
            multiplier *= counts[block_id]
        return coordinates

    def pid_task_for_logical_coordinates(
        root: int,
        coordinates: dict[int, str],
    ) -> str:
        """Invert the root's PID task order without changing its body."""
        axis_order = root_pid_axis_orders[root]
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

    def body_with_nested_loop_waits(
        plan: ReadinessCounterPlan,
        readiness_consumer: ReadinessConsumer,
        body: list[ast.stmt],
        consumer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        assert readiness_consumer.consumer_site_id is not None
        domain = readiness_consumer.keys_by_consumer.source_domain
        nested_axes = nested_logical_axes(
            root_domains[readiness_consumer.consumer_root], domain
        )
        if len(nested_axes) != 1:
            raise AssertionError(
                "nested loop lowering currently requires one loop axis"
            )
        (nested_axis,) = nested_axes
        boundaries = tuple(
            sorted(
                {
                    boundary
                    for piece in readiness_consumer.keys_by_consumer.pieces
                    for axis, begin, end, _step in piece.source_bounds_items
                    if axis == nested_axis
                    for boundary in (begin, end)
                    if 0 < boundary < domain.axis_counts[nested_axis]
                }
            )
        )
        segment_begin_iterations = (0, *boundaries)
        segment_waits: list[tuple[ast.stmt, ...]] = []
        for nested_iteration in segment_begin_iterations:
            site_coordinates = {
                **consumer_coordinates,
                nested_axis: str(nested_iteration),
            }
            readiness_key, membership = relation_flat_target(
                readiness_consumer.keys_by_consumer,
                site_coordinates,
            )
            if membership == "False":
                raise AssertionError("nested-loop segment has no readiness key")
            segment_waits.append(
                tuple(
                    _wait_for_counter(
                        device_function=device_function,
                        counter=readiness_counter(plan, readiness_key),
                        target=(
                            f"tl.cast({epoch_var}, tl.uint32) * "
                            f"tl.cast({readiness_expected_arrivals(plan, readiness_key)}, tl.uint32)"
                        ),
                        prefix="tile_dependency_nested_loop_wait",
                    )
                )
            )
        return _clone_opaque_statements_with_loop_segments(
            body,
            site_id=readiness_consumer.consumer_site_id,
            split_iteration_offsets=boundaries,
            segment_waits=tuple(segment_waits),
        )

    def readiness_counter(
        plan: ReadinessCounterPlan,
        readiness_key: str,
    ) -> str:
        assert readiness_counter_arg is not None
        offset = readiness_counter_offsets[plan]
        offset_text = (
            str(offset)
            if isinstance(offset, int)
            else device_function.sympy_expr(offset)
        )
        return (
            f"{readiness_counter_arg} + {offset_text} + "
            f"({readiness_key}) * {readiness_counter_stride}"
        )

    def readiness_expected_arrivals(
        plan: ReadinessCounterPlan,
        readiness_key: str,
    ) -> str:
        uniform = plan.uniform_arrival_count()
        if uniform is not None:
            return str(uniform)
        readiness_key_coordinates = flat_task_coordinates(
            readiness_key,
            plan.readiness_key_domain.axis_order,
            plan.readiness_key_domain.axis_counts,
        )
        expressions: list[str] = []
        for readiness_producer in plan.producers:
            cardinality = readiness_producer.arrival_count_by_key
            if cardinality is None:
                raise AssertionError("event fan-in is not symbolically known")
            values, _membership = relation_point_coordinates(
                cardinality,
                readiness_key_coordinates,
            )
            expressions.append(values[cardinality.target_domain.axis_order[0]])
        return " + ".join(f"({expression})" for expression in expressions)

    def emit_readiness_arrival_for_key(
        plan: ReadinessCounterPlan,
        readiness_key: str,
    ) -> list[ast.stmt]:
        continuation_consumer = plan.continuation_consumer
        if continuation_consumer is None:
            counter = readiness_counter(plan, readiness_key)
            if plan.uniform_arrival_count() == 1:
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

        converse_consumer = continuation_consumer.keys_by_consumer.converse()
        if converse_consumer is None or not converse_consumer.is_total_function():
            raise AssertionError(
                "a continuation event must bijectively cover its consumer"
            )
        if converse_consumer.is_positional_bijection():
            consumer_task_expression = readiness_key
        else:
            readiness_key_coordinates = flat_task_coordinates(
                readiness_key,
                plan.readiness_key_domain.axis_order,
                plan.readiness_key_domain.axis_counts,
            )
            consumer_coordinates, _membership = relation_point_coordinates(
                converse_consumer,
                readiness_key_coordinates,
            )
            consumer_task_expression = logical_task_from_coordinates(
                continuation_consumer.consumer_root,
                consumer_coordinates,
            )
        consumer_task = device_function.new_var(
            "tile_dependency_continuation_task", dce=True
        )
        assignments = [
            statement_from_string(f"{consumer_task} = {consumer_task_expression}")
        ]
        continuation_root = continuation_consumer.consumer_root
        consumer_coordinates = flat_task_coordinates(
            consumer_task,
            root_domains[continuation_root].axis_order,
            root_axis_counts[continuation_root],
        )
        consumer_logical_pid = (
            f"{case_offsets[continuation_root]} + "
            f"{pid_task_for_logical_coordinates(continuation_root, consumer_coordinates)}"
        )
        consumer_extra_arguments = (consumer_task,)
        previous = device_function.new_var(
            "tile_dependency_continuation_previous", dce=False
        )
        consumer_call = _outline_opaque_tile_body(
            owner,
            device_function,
            root=continuation_root,
            logical_pid=consumer_logical_pid,
            body=body_with_nested_loop_publications(
                continuation_root,
                case_bodies[continuation_root],
                consumer_coordinates,
            ),
            extra_argument_names=consumer_extra_arguments,
        )
        consumer_publications: list[ast.stmt] = []
        for nested_counter, nested_producer in root_counters_by_producer.get(
            continuation_root, ()
        ):
            consumer_publications.extend(
                emit_readiness_arrivals_from_producer(
                    nested_counter,
                    nested_producer,
                    consumer_coordinates,
                )
            )

        last_arrival_body = [consumer_call]
        if consumer_publications:
            last_arrival_body.append(_publication_sync(device_function))
            last_arrival_body.extend(consumer_publications)
        last_arrival_body.extend(root_barrier_publication(continuation_root))
        expected_arrivals = plan.uniform_arrival_count()
        if expected_arrivals is None:
            raise AssertionError(
                "final-arrival continuation requires uniform readiness fan-in"
            )
        if expected_arrivals == 1:
            return [*assignments, *last_arrival_body]
        arrival_counter = readiness_counter(plan, readiness_key)
        return [
            *assignments,
            *_emit_final_arrival_continuation(
                counter=arrival_counter,
                epoch=epoch_var,
                expected_arrivals=expected_arrivals,
                previous=previous,
                continuation_body=last_arrival_body,
            ),
        ]

    def emit_readiness_arrivals_from_producer(
        plan: ReadinessCounterPlan,
        readiness_producer: ReadinessProducer,
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        publication = readiness_producer.keys_by_producer
        if publication is None:
            raise AssertionError("readiness publication relation is unavailable")
        readiness_key, membership = relation_flat_target(
            publication,
            producer_coordinates,
        )
        publications = emit_readiness_arrival_for_key(plan, readiness_key)
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

    def body_with_nested_loop_publications(
        root: int,
        body: list[ast.stmt],
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        """Publish nested-loop readiness without moving the owning root task."""
        site_ids = {
            site_id
            for site_id, readiness_producers in producer_counters_by_site.items()
            if any(
                readiness_producer.producer_root == root
                for _plan, readiness_producer in readiness_producers
            )
        }
        if not site_ids:
            return body
        emitted_site_ids: set[int] = set()

        def rewrite(loop: ast.For) -> list[ast.stmt] | None:
            site_id = tile_dependency_site_id(loop)
            if site_id is None or site_id not in site_ids:
                return None
            if site_id in emitted_site_ids:
                raise AssertionError(
                    "one dependency site must identify one lowered loop"
                )
            readiness_producers = producer_counters_by_site[site_id]
            producer_site_domains = {
                readiness_producer.producers_by_key.target_domain
                for _plan, readiness_producer in readiness_producers
                if readiness_producer.producer_root == root
            }
            if len(producer_site_domains) != 1:
                raise AssertionError(
                    "nested loop publications must share one producer domain"
                )
            (site_domain,) = producer_site_domains
            nested_axes = nested_logical_axes(root_domains[root], site_domain)
            if (
                len(nested_axes) != 1
                or not isinstance(loop.target, ast.Name)
                or not isinstance(loop.iter, ast.Call)
                or len(loop.iter.args) < 2
            ):
                raise AssertionError(
                    "nested loop publication requires one range-like loop axis"
                )
            nested_axis = nested_axes[0]
            begin = ast.unparse(loop.iter.args[0])
            step = ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
            site_coordinates = {
                **producer_coordinates,
                nested_axis: f"(({loop.target.id}) - ({begin})) // ({step})",
            }

            publications: list[ast.stmt] = []
            for plan, readiness_producer in producer_counters_by_site[site_id]:
                publication = readiness_producer.keys_by_producer
                if publication is None:
                    raise AssertionError(
                        "nested-loop readiness publication is unavailable"
                    )
                readiness_key, membership = relation_flat_target(
                    publication,
                    site_coordinates,
                )
                readiness_publications = emit_readiness_arrival_for_key(
                    plan, readiness_key
                )
                if membership == "True":
                    publications.extend(readiness_publications)
                else:
                    publications.append(
                        create(
                            ast.If,
                            test=expr_from_string(membership),
                            body=readiness_publications,
                            orelse=[],
                        )
                    )

            cloned = cast("ast.For", _clone_ast_value(loop))
            cloned.body.extend(
                [
                    _publication_sync(device_function),
                    *publications,
                ]
            )
            emitted_site_ids.add(site_id)
            return [cloned]

        result = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
        if emitted_site_ids != site_ids:
            missing = sorted(site_ids - emitted_site_ids)
            raise AssertionError(f"missing nested producer sites {missing}")
        return result

    def scheduled_logical_task_expression(
        root: int,
        task_order_index: str,
    ) -> str | None:
        """Map one root-local task-order index to its logical task ID."""
        if root not in scheduled_task_roots:
            return None
        traversal = root_schedule_traversals[root]
        forward = traversal.scheduled_ordinal_to_logical_task
        if forward is not None:
            task_order_coordinates = flat_task_coordinates(
                task_order_index,
                forward.source_domain.axis_order,
                forward.source_domain.axis_counts,
            )
            task_coordinates, membership = relation_point_coordinates(
                forward,
                task_order_coordinates,
            )
            if membership != "True":
                raise AssertionError(
                    "proved root traversal is not total during codegen"
                )
            return logical_task_from_coordinates(root, task_coordinates)

        # Some existing PID permutations have no single representable forward
        # relation.  Render the certificate's own segment ranges directly;
        # never reconstruct their boundaries or ordering in codegen.
        expression = ""
        for segment, ordinal_begin, ordinal_end in reversed(
            traversal.segment_ordinal_ranges
        ):
            logical_order = segment.logical_task_order
            if logical_order is None:
                raise AssertionError("segment has no dense rendering order")
            task_order_delta = f"(({task_order_index}) - {ordinal_begin})"
            membership = (
                f"({task_order_delta}) >= 0 and "
                f"({task_order_delta}) < {ordinal_end - ordinal_begin}"
            )
            task_order_coordinates = flat_task_coordinates(
                task_order_delta,
                logical_order.source_domain.axis_order,
                logical_order.source_domain.axis_counts,
            )
            task_coordinates, relation_membership = relation_point_coordinates(
                logical_order,
                task_order_coordinates,
            )
            if relation_membership != "True":
                membership = f"({membership}) and ({relation_membership})"
            segment_task = logical_task_from_coordinates(root, task_coordinates)
            expression = (
                segment_task
                if not expression
                else f"tl.where({membership}, {segment_task}, {expression})"
            )
        if not expression:
            raise AssertionError(f"root {root} has no certified traversal")
        return expression

    def scheduled_root_task_body(
        root: int,
        root_local_pid_task: str,
        logical_pid: str,
        extra_argument_names: tuple[str, ...],
        *,
        force_noinline: bool = False,
    ) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        has_task_scheduling = root in nested_loop_counters_by_consumer
        producer_counters = tuple(root_counters_by_producer.get(root, ()))
        scheduled_pid_task = root_local_pid_task
        scheduled_logical_pid = logical_pid
        scheduled_coordinates: dict[int, str] | None = None
        if producer_counters or root in nested_producer_roots:
            has_task_scheduling = True
        if (
            logical_task_expr := scheduled_logical_task_expression(
                root,
                root_local_pid_task,
            )
        ) is not None:
            scheduled_logical_task = device_function.new_var(
                "tile_dependency_scheduled_logical_task", dce=True
            )
            pid_task = device_function.new_var(
                "tile_dependency_scheduled_pid_task", dce=True
            )
            scheduled_coordinates = flat_task_coordinates(
                scheduled_logical_task,
                root_domains[root].axis_order,
                root_axis_counts[root],
            )
            pid_task_expression = pid_task_for_logical_coordinates(
                root, scheduled_coordinates
            )
            body.extend(
                [
                    statement_from_string(
                        f"{scheduled_logical_task} = {logical_task_expr}"
                    ),
                    statement_from_string(f"{pid_task} = {pid_task_expression}"),
                ]
            )
            scheduled_pid_task = pid_task
            scheduled_logical_pid = f"{case_offset_strings[root]} + {pid_task}"
        if scheduled_coordinates is None:
            scheduled_coordinates = logical_coordinates_for_pid_task(
                root, scheduled_pid_task
            )
        for (
            incoming_readiness_counter,
            incoming_consumer,
        ) in readiness_consumers_by_root.get(root, ()):
            has_task_scheduling = True
            assert readiness_counter_arg is not None
            readiness_key, membership = relation_flat_target(
                incoming_consumer.keys_by_consumer,
                scheduled_coordinates,
            )
            wait = _wait_for_counter(
                device_function=device_function,
                counter=readiness_counter(
                    incoming_readiness_counter,
                    readiness_key,
                ),
                target=(
                    f"tl.cast({epoch_var}, tl.uint32) * "
                    f"tl.cast({readiness_expected_arrivals(incoming_readiness_counter, readiness_key)}, tl.uint32)"
                ),
                prefix="tile_dependency_readiness_wait",
            )
            if not incoming_consumer.keys_by_consumer.is_total_function():
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
        nested_loop_consumers = nested_loop_counters_by_consumer.get(root, ())
        if nested_loop_consumers:
            # Instrument the original logical loop before segmentation.
            # Split ranges then retain the original site-coordinate
            # expression instead of rebasing publication IDs per segment.
            scheduled_root_body = body_with_nested_loop_publications(
                root,
                case_bodies[root],
                scheduled_coordinates,
            )
            for loop_plan, loop_consumer in sorted(
                nested_loop_consumers,
                key=lambda item: (
                    item[1].consumer_site_id
                    if item[1].consumer_site_id is not None
                    else -1
                ),
            ):
                scheduled_root_body = body_with_nested_loop_waits(
                    loop_plan,
                    loop_consumer,
                    scheduled_root_body,
                    scheduled_coordinates,
                )
            tile_body = [
                statement_from_string(
                    f"{owner.shared_pid_var} = {scheduled_logical_pid}"
                ),
                *scheduled_root_body,
            ]
        else:
            tile_body = [
                statement_from_string(
                    f"{owner.shared_pid_var} = {scheduled_logical_pid}"
                ),
                *_clone_opaque_statements(
                    body_with_nested_loop_publications(
                        root,
                        case_bodies[root],
                        scheduled_coordinates,
                    )
                ),
            ]
        if root in kernel_scope_roots:
            body.extend(tile_body)
        else:
            body.append(
                _outline_cross_loop_region(
                    device_function,
                    name_hint=f"tile_dependency_root_{root}",
                    body=tile_body,
                    extra_argument_names=extra_argument_names,
                    noinline=force_noinline,
                )
            )
        if producer_counters:
            has_task_scheduling = True
            body.append(_publication_sync(device_function))
        for producer_counter_plan, readiness_producer in producer_counters:
            body.extend(
                emit_readiness_arrivals_from_producer(
                    producer_counter_plan,
                    readiness_producer,
                    scheduled_coordinates,
                )
            )
        if not has_task_scheduling or root in kernel_scope_roots:
            return body
        # A root used at one schedule occurrence with entry-only bookkeeping
        # has no duplicated call site to protect.  Let Triton inline that thin
        # wrapper just as it does for a root-barrier wait surrounding the same
        # body.  A root-barrier publication is emitted by the caller after the
        # task dispatch, so it does not require an outlining boundary.  Repeated
        # roots and roots that publish per-task counters or splice nested work
        # retain the noinline boundary that limits code growth and live-range
        # coupling across schedule occurrences.
        root_segments = static_pipeline_plan.worker_schedule.segments_for_root(root)
        is_single_trip_occurrence = (
            not has_parameterized_schedule
            and len(root_segments) == 1
            and root_segments[0].task_count <= root_segments[0].worker_count
        )
        scheduled_wrapper_noinline = (
            not is_single_trip_occurrence
            or bool(producer_counters)
            or root in nested_producer_roots
        )
        return [
            _outline_cross_loop_region(
                device_function,
                name_hint=f"tile_dependency_root_{root}_scheduled_task",
                body=body,
                extra_argument_names=extra_argument_names,
                noinline=scheduled_wrapper_noinline,
            )
        ]

    static_segments_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}
    parameterized_schedule_geometry = (
        parameterized_root_major_geometry
        if parameterized_root_major_geometry is not None
        else parameterized_event_frontier_geometry
    )
    parameterized_segment_geometry_by_root = (
        {
            segment.root: (segment, first_wave, task_count)
            for segment, first_wave, task_count in parameterized_schedule_geometry
        }
        if parameterized_schedule_geometry is not None
        else {}
    )
    for root in range(len(root_domains)):
        segments = static_pipeline_plan.worker_schedule.segments_for_root(root)
        if not segments:
            continue
        if parameterized_schedule_geometry is not None:
            if (
                len(segments) != 1
                or root not in parameterized_segment_geometry_by_root
                or parameterized_segment_geometry_by_root[root][0] != segments[0]
            ):
                raise exc.InvalidConfig(
                    "cross_loop_schedule='static_pipeline' does not support "
                    f"parameterized root {root}'s worker assignment"
                )
            static_segments_by_root[root] = segments
            continue
        traversal = root_schedule_traversals.get(root)
        if traversal is None or any(
            segment.dispatch_offset % segment.worker_count for segment in segments
        ):
            raise exc.InvalidConfig(
                "cross_loop_schedule='static_pipeline' does not "
                f"support root {root}'s worker assignment"
            )
        static_segments_by_root[root] = segments

    def worker_membership_condition(
        intervals: tuple[WorkerInterval, ...],
    ) -> str:
        """Render compact membership in symbolic resident-worker intervals."""
        intervals = _normalize_intervals(intervals)
        if not intervals:
            raise AssertionError("an executable segment requires an active worker")
        return " or ".join(
            (
                f"({worker}) == {begin}"
                if end == begin + 1
                else f"(({worker}) >= {begin} and ({worker}) < {end})"
            )
            for begin, end in intervals
        )

    shared_scheduled_task_body_by_root: dict[int, list[ast.stmt]] = {}

    def shared_scheduled_task_body(root: int) -> list[ast.stmt]:
        """Build one reusable task executor for every occurrence of a root."""
        body = shared_scheduled_task_body_by_root.get(root)
        if body is None:
            root_local_pid_task = (
                f"({strategy.virtual_pid_var}) - {case_offset_strings[root]}"
            )
            body = scheduled_root_task_body(
                root,
                root_local_pid_task,
                strategy.virtual_pid_var,
                (strategy.virtual_pid_var,),
            )
            shared_scheduled_task_body_by_root[root] = body
        return _clone_opaque_statements(body)

    def static_segment_body(
        segment: WorkerScheduleSegment,
        *,
        task_order_begin: int | None,
        publish_root_barrier_workers: tuple[WorkerInterval, ...],
    ) -> list[ast.stmt]:
        """Lower one run at its authoritative position in the segment stream."""
        root = segment.root
        if root in continuation_roots:
            raise AssertionError("continuation-owned root has a static segment")
        if segment not in static_segments_by_root.get(root, ()):
            raise AssertionError("static segment is not owned by its root")
        task_dispatch: list[ast.stmt]
        segment_workers = (
            ((0, launch_worker_count),)
            if has_parameterized_schedule
            else segment.worker_intervals()
        )
        if parameterized_root_major_geometry is not None:
            geometry_segment, _first_wave, task_count = (
                parameterized_segment_geometry_by_root[root]
            )
            if segment != geometry_segment or task_order_begin is not None:
                raise AssertionError(
                    "parameterized segment disagrees with its proved relation"
                )
            task_count_text = device_function.sympy_expr(task_count)
            case_offset_text = case_offset_strings[root]
            task_dispatch = [
                create(
                    ast.For,
                    target=create(
                        ast.Name,
                        id=strategy.virtual_pid_var,
                        ctx=ast.Store(),
                    ),
                    iter=expr_from_string(
                        f"tl.range(({case_offset_text}) + ({worker}), "
                        f"({case_offset_text}) + ({task_count_text}), "
                        f"{segment.worker_count})"
                    ),
                    body=shared_scheduled_task_body(root),
                    orelse=[],
                    type_comment=None,
                )
            ]
        elif (
            segment.task_count == 1
            and root_domains[root].size == 1
            and len(static_segments_by_root[root]) == 1
        ):
            task_dispatch = scheduled_root_task_body(
                root,
                "0",
                str(case_offsets[root]),
                (epoch_var,),
                force_noinline=True,
            )
        else:
            assert task_order_begin is not None
            segment_begin = case_offsets[root] + task_order_begin
            segment_end = segment_begin + segment.task_count
            task_dispatch = [
                create(
                    ast.For,
                    target=create(
                        ast.Name,
                        id=strategy.virtual_pid_var,
                        ctx=ast.Store(),
                    ),
                    iter=expr_from_string(
                        f"tl.range((({worker}) - {segment.worker_begin}) + "
                        f"({segment_begin}), ({segment_end}), "
                        f"{segment.worker_count})"
                    ),
                    body=shared_scheduled_task_body(root),
                    orelse=[],
                    type_comment=None,
                )
            ]
        incoming_roots = root_barrier_incoming.get(root, ())
        if (
            segment_workers == ((0, launch_worker_count),)
            and not incoming_roots
            and not publish_root_barrier_workers
        ):
            return task_dispatch

        active_body = _wait_for_dependencies(
            device_function=device_function,
            dependencies=root_barrier_input_dependencies(root),
            prefix="tile_dependency_root_barrier_wait",
        )
        active_body.extend(task_dispatch)
        if publish_root_barrier_workers:
            publications = root_barrier_publication(root)
            if publish_root_barrier_workers == segment_workers:
                active_body.extend(publications)
            else:
                active_body.append(
                    create(
                        ast.If,
                        test=expr_from_string(
                            worker_membership_condition(publish_root_barrier_workers)
                        ),
                        body=publications,
                        orelse=[],
                    )
                )
        return [
            create(
                ast.If,
                test=expr_from_string(worker_membership_condition(segment_workers)),
                body=active_body,
                orelse=[],
            )
        ]

    publication_workers_by_segment = {
        publication.segment_index: publication.worker_intervals
        for publication_plan in root_publication_plans.values()
        for publication in publication_plan.publications
    }

    resident_body: list[ast.stmt] = []
    next_segment_range_by_root: dict[int, int] = {}
    if parameterized_event_frontier_geometry is not None:
        if publication_workers_by_segment:
            raise AssertionError(
                "event-frontier recurrence cannot contain a root barrier"
            )
        task_count = parameterized_event_frontier_geometry[0][2]
        task_count_text = device_function.sympy_expr(task_count)
        frontier_task = device_function.new_var(
            "tile_dependency_event_frontier_task",
            dce=True,
        )
        frontier_body: list[ast.stmt] = []
        for segment, _phase, _task_count in parameterized_event_frontier_geometry:
            root = segment.root
            frontier_body.extend(
                scheduled_root_task_body(
                    root,
                    frontier_task,
                    f"{case_offset_strings[root]} + {frontier_task}",
                    (frontier_task,),
                    force_noinline=True,
                )
            )
        resident_body.append(
            create(
                ast.For,
                target=create(
                    ast.Name,
                    id=frontier_task,
                    ctx=ast.Store(),
                ),
                iter=expr_from_string(
                    f"tl.range({worker}, {task_count_text}, {launch_worker_count})"
                ),
                body=frontier_body,
                orelse=[],
                type_comment=None,
            )
        )
    else:
        for segment_index, segment in enumerate(
            static_pipeline_plan.worker_schedule.segments
        ):
            root = segment.root
            if root == transient_source_root:
                continue
            if parameterized_root_major_geometry is not None:
                resident_body.extend(
                    static_segment_body(
                        segment,
                        task_order_begin=None,
                        publish_root_barrier_workers=(
                            publication_workers_by_segment.get(segment_index, ())
                        ),
                    )
                )
                continue
            range_index = next_segment_range_by_root.get(root, 0)
            ranges = root_schedule_traversals[root].segment_ordinal_ranges
            if range_index >= len(ranges):
                raise AssertionError("segment stream exceeds its proved root traversal")
            certified_segment, task_order_begin, task_order_end = ranges[range_index]
            if (
                certified_segment != segment
                or task_order_end - task_order_begin != segment.task_count
            ):
                raise AssertionError(
                    "segment stream disagrees with its proved traversal"
                )
            next_segment_range_by_root[root] = range_index + 1
            resident_body.extend(
                static_segment_body(
                    segment,
                    task_order_begin=task_order_begin,
                    publish_root_barrier_workers=publication_workers_by_segment.get(
                        segment_index, ()
                    ),
                )
            )
    if any(
        root != transient_source_root
        and next_segment_range_by_root.get(root, 0)
        != len(traversal.segment_ordinal_ranges)
        for root, traversal in root_schedule_traversals.items()
    ):
        raise AssertionError("proved root traversal has unconsumed segments")
    if transient_source_root is None:
        result.extend(resident_body)
        result.append(
            statement_from_string(f"tl.store({epoch_arg} + {worker}, {epoch_var})")
        )
    else:
        assert dispatch_ticket is not None
        if (
            transient_source_root in readiness_consumers_by_root
            or transient_source_root in root_barrier_incoming
        ):
            raise AssertionError(
                "a transient source may not have incoming dependencies"
            )
        transient_body = scheduled_root_task_body(
            transient_source_root,
            dispatch_ticket,
            f"{case_offsets[transient_source_root]} + {dispatch_ticket}",
            (dispatch_ticket,),
        )
        transient_body.extend(root_barrier_publication(transient_source_root))
        if kernel_scope_roots - {transient_source_root}:
            resident_branch = resident_body
        else:
            resident_branch = [
                _outline_cross_loop_region(
                    device_function,
                    name_hint="tile_dependency_resident_worker",
                    body=resident_body,
                    extra_argument_names=(worker, epoch_var),
                    noinline=True,
                )
            ]
        result.append(
            create(
                ast.If,
                test=expr_from_string(
                    f"{dispatch_ticket} < {transient_source_task_count}"
                ),
                body=transient_body,
                orelse=resident_branch,
            )
        )
    if (
        tuple(_ast_fingerprint(body) for body in case_bodies)
        != opaque_case_fingerprints
    ):
        raise AssertionError(
            "tile-dependency lowering mutated an opaque source tile body"
        )
    return result
