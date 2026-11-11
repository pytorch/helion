from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import TypeGuard
from typing import cast

import sympy
import torch
from torch.utils._sympy.functions import CeilDiv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

from .. import exc
from .._dist_utils import _resolve_process_group
from ..language import memory_ops
from ..language.atomic_ops import ATOMIC_OPS
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cross_loop_scheduler import CrossLoopDispatchMode
from .cross_loop_scheduler import ReadinessConsumer
from .cross_loop_scheduler import ReadinessCounterPlan
from .cross_loop_scheduler import ReadinessProducer
from .cross_loop_scheduler import _root_barrier_reachability
from .cross_loop_scheduler import build_static_pipeline_plan
from .cross_loop_scheduler import nested_wait_placement
from .device_function import SyncRole
from .device_function import TensorArg
from .device_function import TensorDescriptorArg
from .host_function import HostFunction
from .program_id import _clone_ast_value
from .program_id import _clone_stmt
from .program_id import typed_program_id
from .tile_dependency import _INBAND_PUSH_BYTES
from .tile_dependency import TILE_ACCESS_META
from .tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DenseTaskOrder
from .tile_dependency import TileDependencyKind
from .tile_dependency import _access_interval_expression
from .tile_dependency import _concrete_integer
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_coordinate_domains
from .tile_dependency import nested_logical_axes
from .tile_dependency import tile_dependency_site_id
from .tile_strategy import L2GroupingProgramIDs
from .tile_strategy import TileStrategy
from .triton.prefetch import PREFETCH_ASMS

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .device_function import DeviceFunction
    from .program_id import ForEachProgramID
    from .program_id import PersistentProgramIDs
    from .program_id import PIDInfo
    from .program_id import ProgramIDs
    from .tile_dependency import TileAccess
    from .tile_dependency import TileDependencyGraph


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


# One 128-byte line per uint64 peer_counter slot.
_PEER_SLOT_WORDS = _CROSS_LOOP_COUNTER_ALIGNMENT_BYTES // torch.uint64.itemsize


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
        numel: int | sympy.Expr
        if isinstance(numel_expr, int):
            numel = numel_expr
        elif isinstance(numel_expr, torch.SymInt):
            numel = cast("sympy.Expr", numel_expr._sympy_())
        elif getattr(numel_expr, "is_number", False):
            numel = int(numel_expr)
        elif isinstance(numel_expr, sympy.Expr):
            numel = numel_expr
        else:
            return None
        if isinstance(numel, sympy.Expr):
            numel = env.specialize_expr(numel)
            if numel.is_number:
                numel = int(numel)
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
            else cast("sympy.Expr", CeilDiv(numel, sympy.Integer(block)))
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
    info = CompileEnvironment.current().block_sizes[block_id]
    # No static extent (data-dependent bounds, reused block sizes): no geometry.
    if not isinstance(info.size, (int, torch.SymInt)):
        return None
    try:
        numel_expr = info.numel
        if not numel_expr.is_number:
            return None
        numel = int(numel_expr)
        block = int(info.from_config_assert(device_function.config))
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
    if not isinstance(first_count, int | sympy.Integer) or not isinstance(
        second_count, int | sympy.Integer
    ):
        return None
    first_count = int(first_count)
    second_count = int(second_count)
    if second_count == 1 or case.group_size >= first_count:
        return None
    return case.group_size


def _root_task_orders(
    owner: ForEachProgramID,
    root_domains: tuple[CoordinateDomain, ...],
    case_geometries: tuple[
        tuple[tuple[int, ...], dict[int, int | sympy.Expr], dict[int, int]], ...
    ],
) -> tuple[DenseTaskOrder, ...] | None:
    """Bind each logical root domain to its configured PID task order."""
    if len(root_domains) != len(case_geometries):
        return None
    result: list[DenseTaskOrder] = []
    for root, (domain, geometry) in enumerate(
        zip(root_domains, case_geometries, strict=True)
    ):
        pid_axis_order, axis_counts, block_sizes = geometry
        if domain.parameter_symbols and isinstance(
            owner.cases[root], L2GroupingProgramIDs
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
        order = DenseTaskOrder.from_pid(
            domain, pid_axis_order, l2_group_size=l2_group_size
        )
        if order is None:
            return None
        result.append(order)
    return tuple(result)


def _wait_for_counter(
    *,
    device_function: DeviceFunction,
    counter: str,
    target: str,
    prefix: str,
    load_fence: bool = True,
) -> list[ast.stmt]:
    value = device_function.new_var(prefix, dce=False)
    sync = device_function.new_var(f"{prefix}_sync", dce=False)
    warp_sync = (
        f"{sync} = tl.inline_asm_elementwise("
        "asm='bar.warp.sync 0xffffffff; mov.u32 $0, $1;', "
        "constraints='=r,r', args=[VEHICLE], dtype=tl.uint32, is_pure=False, pack=1)"
    )
    scalar = warp_sync.replace("VEHICLE", "tl.full([], 0, tl.int32)")
    device_function.sync_roles[value] = SyncRole("wait")
    device_function.sync_roles[sync] = SyncRole(
        "wait", scalar=statement_from_string(scalar)
    )
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
        statement_from_string(warp_sync.replace("VEHICLE", "tl.arange(0, 32)")),
        *(device_function.async_load_fence() if load_fence else []),
    ]


def _covering(
    device_function: DeviceFunction,
    wait: list[ast.stmt],
    allocations: frozenset[int],
    publication: bool = False,
) -> list[ast.stmt]:
    """``wait``, recorded as ordering reads of ``allocations`` only (and, with
    ``publication``, its root's barrier publication).  An allocation no tensor
    argument names is one no partition can tell it reads: the wait stays
    ordering every access."""
    if not allocations <= set(_argument_allocations(device_function).values()):
        return wait
    for name in _stored_names(wait) & device_function.sync_names("wait"):
        role = device_function.sync_roles[name]
        role.covers, role.publication = allocations, publication
    return wait


def _covering_as(
    device_function: DeviceFunction, wait: list[ast.stmt], waits: list[ast.stmt]
) -> list[ast.stmt]:
    """``wait`` (e.g. the fence after ``waits``), recorded as ordering what
    ``waits`` together order."""
    roles = [
        device_function.sync_roles[name]
        for name in _stored_names(waits) & device_function.sync_names("wait")
    ]
    if not roles or any(role.covers is None for role in roles):
        return wait
    return _covering(
        device_function,
        wait,
        frozenset().union(*(cast("frozenset[int]", role.covers) for role in roles)),
        any(role.publication for role in roles),
    )


def _wait_for_dependencies(
    *,
    device_function: DeviceFunction,
    dependencies: tuple[tuple[str, str], ...],
    prefix: str,
) -> list[ast.stmt]:
    """Emit every acquire wait in one graph-derived dependency set."""
    waits = [
        statement
        for counter, target in dependencies
        for statement in _wait_for_counter(
            device_function=device_function,
            counter=counter,
            target=target,
            prefix=prefix,
            load_fence=False,
        )
    ]
    return [*waits, *device_function.async_load_fence()] if waits else []


@dataclasses.dataclass(frozen=True)
class _Arrival:
    """One +1 readiness arrival; with ``target``, a final-arrival election that
    runs ``body`` on the arrival that brings the counter to ``target``."""

    counter: str
    membership: str = "True"
    target: str | None = None
    previous: str | None = None
    body: tuple[ast.stmt, ...] = ()


@dataclasses.dataclass(frozen=True)
class _Inline:
    """A continuation of one expected arrival: its consumer task (with the
    consumer's own publications) runs in place of the arrival, as one unit."""

    body: tuple[ast.stmt, ...]
    membership: str = "True"


def _both(first: str, second: str) -> str:
    if first == "True":
        return second
    if second == "True":
        return first
    return f"({first}) and ({second})"


def _guarded(membership: str, body: list[ast.stmt]) -> list[ast.stmt]:
    if membership == "True":
        return body
    return [create(ast.If, test=expr_from_string(membership), body=body, orelse=[])]


def _arrival_parts_under(
    parts: list[ast.Assign | _Arrival | _Inline], membership: str
) -> list[ast.Assign | _Arrival | _Inline]:
    """Guard a producer's arrivals by its membership (the consumer-task index
    assignments stay unguarded: they only compute)."""
    return [
        part
        if isinstance(part, ast.Assign)
        else dataclasses.replace(part, membership=_both(membership, part.membership))
        for part in parts
    ]


def _emit_arrivals(
    device_function: DeviceFunction,
    parts: list[ast.Assign | _Arrival | _Inline],
    sem: str,
) -> list[ast.stmt]:
    """Statements for one publication point's arrivals.

    Each release atomic lowers to a GPU-scope fence plus the atomic, and a
    second fence waits for the first atomic's acknowledgment: a task arriving on
    several counters paid an L2 round trip per extra counter.  The +1 arrivals
    share one atomic over a vector of their counters instead; every lane keeps
    the release, and an election reads its own lane's previous value.  Inline
    continuations run after the arrivals, so other consumers start first.
    """
    assignments = [part for part in parts if isinstance(part, ast.Assign)]
    arrivals = [part for part in parts if isinstance(part, _Arrival)]
    inlines = [part for part in parts if isinstance(part, _Inline)]
    # An election runs its continuation, which reads what every arrival released.
    elect_sem = "acq_rel" if sem == "release" else "acquire"

    def elected(arrival: _Arrival, previous: str) -> ast.stmt:
        return create(
            ast.If,
            test=expr_from_string(f"{previous} == {arrival.target} - 1"),
            body=list(arrival.body),
            orelse=[],
        )

    result: list[ast.stmt] = list(assignments)
    if len(arrivals) == 1:
        (arrival,) = arrivals
        if arrival.target is None:
            atomic = f"tl.atomic_add({arrival.counter}, 1, sem='{sem}', scope='gpu')"
            result.extend(_guarded(arrival.membership, [statement_from_string(atomic)]))
        else:
            assert arrival.previous is not None
            election = [
                statement_from_string(
                    f"{arrival.previous} = tl.atomic_add({arrival.counter}, 1, "
                    f"sem='{elect_sem}', scope='gpu')"
                ),
                elected(arrival, arrival.previous),
            ]
            result.extend(_guarded(arrival.membership, election))
    elif arrivals:
        lanes = device_function.new_var("tile_dependency_arrival_lane", dce=True)
        previous = device_function.new_var(
            "tile_dependency_arrival_previous", dce=False
        )
        counters = f"({arrivals[-1].counter})"
        for index in reversed(range(len(arrivals) - 1)):
            counters = (
                f"tl.where({lanes} == {index}, {arrivals[index].counter}, {counters})"
            )
        if all(arrival.membership == "True" for arrival in arrivals):
            mask = f"{lanes} < {len(arrivals)}"
        else:
            mask = " | ".join(
                f"(({lanes} == {index}) & ({arrival.membership}))"
                for index, arrival in enumerate(arrivals)
            )
        elections = any(arrival.target is not None for arrival in arrivals)
        result.extend(
            [
                statement_from_string(
                    f"{lanes} = tl.arange(0, {1 << (len(arrivals) - 1).bit_length()})"
                ),
                statement_from_string(
                    f"{previous} = tl.atomic_add({counters}, 1, mask={mask}, "
                    f"sem='{elect_sem if elections else sem}', scope='gpu')"
                ),
            ]
        )
        for index, arrival in enumerate(arrivals):
            if arrival.target is None:
                continue
            assert arrival.previous is not None
            result.append(
                statement_from_string(
                    f"{arrival.previous} = tl.sum(tl.where({lanes} == {index}, "
                    f"{previous}, 0))"
                )
            )
            result.extend(
                _guarded(arrival.membership, [elected(arrival, arrival.previous)])
            )
    for inline in inlines:
        result.extend(_guarded(inline.membership, list(inline.body)))
    return result


def _write_guard(body: list[ast.stmt]) -> ast.If | None:
    """The task body's trailing ``if`` (with no else) when the statements before
    it only load and compute, so every write of the task is inside it."""
    *head, last = body
    if not isinstance(last, ast.If) or any(
        not isinstance(statement, ast.Pass) for statement in last.orelse
    ):
        return None
    for call in (
        n for s in [*head, last.test] for n in ast.walk(s) if isinstance(n, ast.Call)
    ):
        if _may_write(call):
            return None
    return last


def _may_write(call: ast.Call) -> bool:
    """Whether a call may write memory: loads (pointer or descriptor) only read,
    any other effect may write."""
    return not _is_load(call) and _effect_kind(call) != "pure"


def _allocation(tensor: torch.Tensor) -> int:
    """The allocation a (fake) tensor views: a peer view stands for its owner's."""
    state = HostFunction.current().compiler_state
    source = state.ssa_source(tensor)
    storage = (tensor if source is None else source).untyped_storage()
    owner = state.peer_views.get(storage)
    return id(owner[0] if owner else storage)


def _access_allocations() -> dict[int, int]:
    """The allocation (``_allocation``) of each tile access of the device IR."""
    return {
        access_id: _allocation(node.args[0].meta["val"])
        for info in HostFunction.current().device_ir.graphs
        for node in info.graph.nodes
        for access_id in node.meta.get(TILE_ACCESS_META, ())
    }


def _written_allocations() -> set[int]:
    """The allocations this kernel stores to or atomically updates, on any rank,
    from the device IR (so also in roots the codegen outlines)."""
    return {
        _allocation(node.args[0].meta["val"])
        for info in HostFunction.current().device_ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store or node.target in ATOMIC_OPS
    }


def _argument_allocations(device_function: DeviceFunction) -> dict[str, int]:
    return {
        argument.name: _allocation(argument.fake_value)
        for argument in device_function.arguments
        if isinstance(argument, TensorArg)
    }


def _guard_reads(
    body: list[ast.stmt], allocations: dict[str, int]
) -> frozenset[int] | None:
    """The allocations loaded before (and by) the trailing write guard (see
    ``_loaded_allocations``)."""
    *head, last = body
    assert isinstance(last, ast.If)
    return _loaded_allocations([*head, last.test], allocations)


def _loaded_allocations(
    statements: Sequence[ast.AST], allocations: dict[str, int]
) -> frozenset[int] | None:
    """The allocations ``statements`` load, given the tensor arguments'
    (descriptors included), or ``None`` when a load's address names no tensor
    argument or inline asm takes one."""
    loaded: set[int] = set()
    for call in (n for s in statements for n in ast.walk(s) if isinstance(n, ast.Call)):
        if _is_load(call):
            # ``tl.*``: the pointer or descriptor is the first argument.
            tl_load = _is_tl_call(call, None)
            address = call.args[0] if tl_load and call.args else call.func.value  # pyrefly: ignore [missing-attribute]
            names = _all_names([address]) & allocations.keys()
            if not names:
                return None
            loaded.update(allocations[name] for name in names)
        elif _is_tl_call(call, frozenset({"inline_asm_elementwise"})) and (
            _all_names([call]) & allocations.keys()
        ):
            return None
    return frozenset(loaded)


def _named_allocations(statement: ast.AST, allocations: dict[str, int]) -> set[int]:
    """The allocations of the tensor arguments ``statement`` names."""
    return {allocations[n] for n in _all_names([statement]) if n in allocations}


def _batch_waits(
    device_function: DeviceFunction, body: list[ast.stmt]
) -> list[ast.stmt]:
    """``body`` with each run of adjacent dependency spins polled together:
    every counter's first load ahead of any spin (and adjacent wait-only
    guards of one test merged), so waits already met cost one memory round
    trip between them rather than one each.  Each spin still precedes the
    reads after the run."""
    waits = device_function.sync_names("wait")

    def asm(statement: ast.stmt) -> str:
        call = statement.value if isinstance(statement, ast.Assign) else None
        if not (isinstance(call, ast.Call) and call.keywords):
            return ""
        text = next((k.value for k in call.keywords if k.arg == "asm"), None)
        return cast("str", text.value) if isinstance(text, ast.Constant) else ""

    def spin(index: int) -> int:
        """The length of the spin at ``index`` (``_wait_for_counter``'s load,
        while loop, warp sync and optional fence), else 0."""
        if index + 2 >= len(body):
            return 0
        load, loop, sync = body[index : index + 3]
        target = _stored_names([load])
        if not (
            isinstance(load, ast.Assign)
            and len(target) == 1
            and target <= waits
            and "ld.acquire" in asm(load)
            and isinstance(loop, ast.While)
            and _all_names([loop.test]) & target
            and "bar.warp.sync" in asm(sync)
        ):
            return 0
        fence = index + 3 < len(body) and asm(body[index + 3]).startswith(
            "fence.proxy.async.global; mov"
        )
        return 3 + fence

    def guard(statement: ast.stmt) -> ast.If | None:
        """A guard holding spins only, whose test they do not feed and whose
        test is pure (merged guards evaluate it once)."""
        if not (isinstance(statement, ast.If) and not statement.orelse):
            return None
        names = _stored_names(statement.body)
        if not names or not names <= waits or _all_names([statement.test]) & names:
            return None
        if any(
            _effect_kind(call) != "pure"
            for call in ast.walk(statement.test)
            if isinstance(call, ast.Call)
        ):
            return None
        return statement

    kept: list[ast.stmt] = []
    index = 0
    while index < len(body):
        if length := spin(index):
            spins: list[list[ast.stmt]] = []
            while length:
                spins.append(body[index : index + length])
                index += length
                length = spin(index)
            kept += [s[0] for s in spins]
            kept += [s[1] for s in spins]
            kept += [t for s in spins for t in s[2:]]
            continue
        statement = body[index]
        if (first := guard(statement)) is not None:
            merged = list(first.body)
            while (
                index + 1 < len(body)
                and (other := guard(body[index + 1])) is not None
                and ast.dump(other.test) == ast.dump(first.test)
            ):
                merged += other.body
                index += 1
            statement = create(
                ast.If,
                test=first.test,
                body=_batch_waits(device_function, merged),
                orelse=[],
            )
        kept.append(statement)
        index += 1
    return kept


def _scope_wait(
    body: list[ast.stmt],
    wait: list[ast.stmt],
    covered: frozenset[int],
    state: frozenset[str],
    allocations: dict[str, int],
) -> None:
    """Insert ``wait`` before the first statement of ``body`` that may touch
    what it orders: one naming a tensor of ``covered`` (allocations) or the
    schedule's ``state``, assigning what the wait reads, or making a call other
    than a pure op or a load from a tensor argument plus pure index math."""
    reads = _all_names(wait)

    def independent_call(call: ast.Call) -> bool:
        if not _is_tl_call(call, frozenset({"load"})):
            return _effect_kind(call) == "pure"
        return bool(
            call.args and _named_allocations(call.args[0], allocations)
        ) and all(
            _effect_kind(c) == "pure"
            and not (
                isinstance(c.func, ast.Attribute) and c.func.attr == "pointer_type"
            )
            for c in ast.walk(call.args[0])
            if isinstance(c, ast.Call)
        )

    def independent(statement: ast.stmt) -> bool:
        assigned = _stored_names([statement])
        return not (
            _named_allocations(statement, allocations) & covered
            or _all_names([statement]) & state
            or assigned & reads
        ) and all(
            independent_call(call)
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        )

    index = next((i for i, s in enumerate(body) if not independent(s)), len(body))
    body[index:index] = wait


def _bound_wait(device_function: DeviceFunction, call: str) -> ast.stmt:
    """A wait the scheduler emits as a call, bound to a name with a wait role
    (as every scheduler wait is) so that later lowerings recognize it."""
    token = device_function.new_var("tile_dependency_wait", dce=False)
    device_function.sync_roles[token] = SyncRole("wait")
    return statement_from_string(f"{token} = {call}")


def _publication_sync(device_function: DeviceFunction) -> ast.stmt:
    if cast("int", device_function.config.get("num_warps", 1)) != 1:
        return statement_from_string("tl.debug_barrier()")
    sync = device_function.new_var("tile_dependency_publication_sync", dce=False)
    device_function.sync_roles[sync] = SyncRole("release")
    return statement_from_string(
        f"{sync} = tl.inline_asm_elementwise("
        "asm='bar.warp.sync 0xffffffff; mov.u32 $0, $1;', "
        "constraints='=r,r', args=[tl.arange(0, 32)], "
        "dtype=tl.uint32, is_pure=False, pack=1)"
    )


def _release_sync(device_function: DeviceFunction) -> list[ast.stmt]:
    """CTA sync before a release publication, completing TMA stores first."""
    return [
        *device_function.async_store_drain(),
        _publication_sync(device_function),
    ]


def _register_cross_loop_state(
    device_function: DeviceFunction,
    *,
    name_hint: str,
    numel: str,
    dtype: torch.dtype,
    symmetric: bool = False,
) -> str:
    """Register launch-persistent global state owned by the Triton launcher.

    A symmetric state is followed by the table of every rank's state pointer.
    """
    like = next(
        (
            argument
            for argument in device_function.arguments
            if isinstance(argument, TensorArg)
            and not isinstance(argument, TensorDescriptorArg)
            and argument._host_str is not None
        ),
        None,
    )
    if like is None:
        descriptor = next(
            argument
            for argument in device_function.arguments
            if isinstance(argument, TensorDescriptorArg)
        )
        like_host = (
            HostFunction.current().tensor_to_origin[descriptor.fake_value].host_str()
        )
    else:
        like_host = like.host_str()
    name = device_function.new_var(name_hint, dce=False)
    names = [name]
    if symmetric:
        names.append(device_function.new_var(f"{name_hint}_ptrs", dce=False))
    device_function.wrapper_only_params.extend(names)
    device_function.triton_persistent_state_args.extend(names)
    device_function.triton_persistent_state_specs.append(
        (like_host, numel, str(dtype), symmetric)
    )
    return name


@dataclasses.dataclass(frozen=True)
class ScatterRoot:
    """Tasks of a root with inband scatter stores, each owing one done word."""

    axis_order: tuple[int, ...]
    counts: dict[int, int]
    tasks: int
    # [parity][source][task] words, pushed by the root's last scatter store.
    done: int
    last: int

    def task(self, coordinates: Mapping[int, str]) -> str:
        """Flat task index of per-axis tile coordinates."""
        terms, multiplier = [], 1
        for block_id in self.axis_order:
            if self.counts[block_id] != 1:
                terms.append(f"({coordinates[block_id]}) * {multiplier}")
            multiplier *= self.counts[block_id]
        return " + ".join(terms) or "0"


@dataclasses.dataclass(frozen=True)
class Scatter:
    """An inband row scatter: which keys (aligned row blocks) each task wrote."""

    root: int
    position: int
    lanes: int
    shape: tuple[int, ...]
    extents: tuple[int, ...]
    key_strides: tuple[int, ...]
    keys: int
    # Producer-local [parity][task][lane] records; consumer-local [source][key].
    records: int
    cover: int


# -- tagged peer transport ---------------------------------------------------
# Tagged peer transport for tiles a producer root pushes into every rank's copy.
#
# A keyed peer counter orders such a push before a later root reads its own copy:
# each producer task stores its tile to every rank and then release-adds a key's
# counter on every rank, which consumers acquire-poll. The release is a
# system-scope fence, which waits for every NVLink write the GPU has in flight,
# not just the task's, so the hand-off costs microseconds under load.
#
# Tagged transport instead stores every 4-byte group of the tile together with the
# launch epoch, ``epoch << 32 | payload``, as one single-copy-atomic 8-byte word
# into a mirror of the tensor in each destination rank's peer state (two mirrors,
# by the epoch's parity), and drops the counter with its publishes, skip marks and
# spins. Data and readiness then arrive together: the consumer's ring producer
# (``gluon_ring.ring_put_tagged``) polls its step's words until their tags are
# the epoch and stores the payloads into the ring slot in place of a TMA copy.
#
# Eligibility:
#
# * T1-T3 and T5 of ``tile_dependency._tagged_failure`` on the graph: one push
#   per task to every rank, a dense tensor all accesses share, one later
#   consumer root, the parity credit;
# * here, a native Gluon ring kernel (``Config.ring_depth``), whose producers poll
#   the words;
# * in ``peer_state``: T7, each rank pushing one share of the tensor (the
#   launcher checks at the first launch that the ranks' shares differ), and
#   mirrors within the inband push budget;
# * T4 once the roots' guards are known: a push in a branch is in its root's
#   guard, whose skipped tasks mark their tiles' presence words;
# * T6 as the consumer's loads are generated: each is a ring operand that a
#   ring producer fills (``gluon_ring``), whole mirror words tiling the tensor;
# * in ``gluon_ring``: the pollers' registers come from what the body leaves.
#
# A failed rule regenerates the kernel with keyed peer counters
# (``generate_ast``).  The push still stores the tile to every rank's copy, after
# its words and unordered (nothing in the launch reads the copy, and no rank
# waits for those stores): the copy holds what peer counters leave once the
# pushing ranks' launches complete, which a later reader on another rank must
# order itself after.  Inside the launch, rows no rank pushes read as zeros,
# where peer counters read the copy's earlier contents (exact semantics cost a
# dependent round trip per step).
#
# A masked push writes only some words, so its consumer cannot poll whole tiles.
# The mask must hold per row of the producer tile (one row block: the whole
# second-to-last dimension, at most 32 rows), and the push also stores a row
# presence word per tile, ``epoch << 32 | rows written``, after the mirror's
# words. The consumer (whose tile spans the whole row block too) reads its tiles'
# presence words and polls only the rows their producers wrote; rows they did not
# write read as zeros. The data itself then says which words to wait for, so a
# poll never waits on a word nobody writes, whatever the masks on the producer
# ranks. A push a guard may skip has presence words too: a skipped task stores
# ``epoch << 32`` (no rows) into its tile's, so the consumer reads zeros there
# whatever the guards' bounds on each rank.


class TaggedPeerFallback(exc.Base):
    """A tagged-transport precondition failed during codegen: regenerate the kernel
    with keyed peer counters."""

    def report(self) -> str:
        return f"tagged peer transport fallback: {self}"


def tagged_allocations(device_function: DeviceFunction) -> frozenset[int]:
    """The allocations this config sends as tagged words."""
    if device_function.tagged_allocation_ids is None:
        device_function.tagged_allocation_ids = frozenset(
            _select_tagged(device_function)
        )
    return device_function.tagged_allocation_ids


def tagged_codegen_graph(device_function: DeviceFunction) -> TileDependencyGraph | None:
    """The dependency graph whose transport is "tagged" for the enabled ones."""
    if device_function.codegen_dependency_graph is None:
        graph = HostFunction.current().device_ir.tile_dependency_graph
        if graph is None:
            return None
        tagged = tagged_allocations(device_function)
        device_function.codegen_dependency_graph = (
            dataclasses.replace(graph, tagged_allocation_ids=tagged)
            if tagged
            else graph
        )
    return device_function.codegen_dependency_graph


def _select_tagged(device_function: DeviceFunction) -> list[int]:
    graph = HostFunction.current().device_ir.tile_dependency_graph
    if (
        graph is None
        or not graph.tagged_candidate_ids
        or CompileEnvironment.current().tagged_peer_disabled
        # A ring producer polls the words into its slots, in a native Gluon ring
        # megakernel (where ``gluon_ring`` sizes the pollers' warps and registers).
        or device_function.ring is None
    ):
        return []
    return sorted(graph.tagged_candidate_ids)


def tagged_coverage_failure(
    graph: TileDependencyGraph,
    tagged: frozenset[int],
    guarded: Collection[int],
) -> str | None:
    """Rule T4: every producer task pushes its tile or, skipped by its root's
    guard (``guarded``: roots with one), marks the tile's row presence words
    (``skipped_task_dispatch``), so no poll waits on a word nobody writes in
    the launch, whatever the guards' bounds on each rank.

    A push in a branch (T1 allows one under the root) must then be in the
    root's guard: ``_guarded_extent`` parses one only as the root's last
    statement after pure prefix statements, so it encloses every push.
    """
    for allocation_id in sorted(tagged):
        stores = graph.tagged_stores(allocation_id)
        if stores[0].root not in guarded and not all(
            graph.runs_at_root(store) for store in stores
        ):
            load = graph.tagged_loads(allocation_id)[0]
            name = load.tensor_name or f"allocation {allocation_id}"
            return f"T4: {name}: a push under a branch that is not the root's guard"
    return None


@dataclasses.dataclass(frozen=True)
class TaggedMirror:
    """A tagged allocation's mirrors in the peer state: two of ``words`` from
    word ``offset`` (by epoch parity), ``lanes`` elements a word; for a masked
    push or one a guard may skip, its row presence words after the payload
    words, from ``presence``: one per producer tile of ``rows`` rows of
    ``chunk`` words, ``chunks`` tiles per row block.  ``share``: the share of
    the tensor this rank pushes (T7)."""

    offset: int
    words: int
    lanes: int
    pair: bool
    share: int = 0
    presence: int | None = None
    chunk: int = 0
    rows: int = 0
    chunks: int = 0

    def presence_word(self, offset: str) -> str:
        """The row presence word of the producer tile whose first element is
        element ``offset`` of the tensor."""
        word = f"({offset} // {self.lanes})"
        return (
            f"{self.presence} + {word} // {self.chunk * self.rows * self.chunks} * "
            f"{self.chunks} + {word} % {self.chunk * self.chunks} // {self.chunk}"
        )


@dataclasses.dataclass(frozen=True)
class PeerState:
    """Symmetric uint64 state: inband mailboxes, peer counters and a done slot.

    Laid out from the dependency graph alone, so body codegen and the
    schedule agree on it whichever registers it first.
    """

    state: str
    epoch: str
    ptrs: str
    rank: str
    bases: tuple[str, ...]
    # allocation id -> (first word, words per slot, elements per word, whether
    # stores may write 16-byte word pairs)
    mailboxes: dict[int, tuple[int, int, int, bool]]
    slots: dict[int, int]
    done: int
    scatters: dict[int, Scatter] = dataclasses.field(default_factory=dict)
    scatter_roots: dict[int, ScatterRoot] = dataclasses.field(default_factory=dict)
    # Tagged allocation id -> its mirrors.
    mirrors: dict[int, TaggedMirror] = dataclasses.field(default_factory=dict)
    # Set by a consumer that read an unwritten row of this rank; launch starts.
    flag: int = 0
    started: int = 0
    # One word per worker for CTA votes in noinline helpers.
    votes: int = 0

    def mailbox(self, allocation_id: int, source: object) -> str:
        """Word offset of ``source``'s slot for this launch's parity."""
        offset, words, _, _ = self.mailboxes[allocation_id]
        return (
            f"{offset} + tl.cast({self.epoch} & 1, tl.int64) * "
            f"{len(self.bases) * words} + {source} * {words}"
        )

    def mirror(self, allocation_id: int) -> str:
        """Word offset of this launch's mirror of a tagged allocation."""
        mirror = self.mirrors[allocation_id]
        return f"{mirror.offset} + tl.cast({self.epoch} & 1, tl.int64) * {mirror.words}"

    def every_rank_store(
        self, device_function: DeviceFunction, offset: str, value: str, threads: int
    ) -> ast.stmt:
        """Store the tagged word ``value`` at word ``offset`` of every rank's
        state (relaxed, system scope) from one thread of ``threads``: through
        the hoisted bases (reloading the base table put an L2 round trip ahead
        of each store) and over a lane per thread (a shorter lane vector
        replicates over the threads, each copy storing again).  A publication
        (``SyncRole`` release)."""
        world = len(self.bases)
        stores = " ".join(
            f"@p st.relaxed.sys.global.u64 [${i + 1}], ${world + 1};"
            for i in range(world)
        )
        addresses = ", ".join(f"{base} + ({offset})" for base in self.bases)
        stored = device_function.new_var("tile_dependency_every_rank", dce=False)
        device_function.sync_roles[stored] = SyncRole("release")
        return statement_from_string(
            f"{stored} = tl.inline_asm_elementwise("
            f"'{{ .reg .pred p; setp.ne.b32 p, ${world + 2}, 0; {stores} "
            f"mov.u32 $0, 0; }}', '=r{',l' * (world + 1)},r', "
            f"[{addresses}, tl.cast({value}, tl.uint64), "
            f"(tl.arange(0, {threads}) == 0).to(tl.int32)], "
            "dtype=tl.int32, is_pure=False, pack=1)"
        )


def peer_state(device_function: DeviceFunction) -> PeerState | None:
    """Register the peer state once per kernel if any dependency crosses ranks."""
    if device_function.peer_state is not None:
        return device_function.peer_state
    graph = tagged_codegen_graph(device_function)
    if graph is None or not graph.crosses_ranks():
        return None
    process_group_name = CompileEnvironment.current().process_group_name
    assert process_group_name is not None
    world_size = torch.distributed.get_world_size(
        _resolve_process_group(process_group_name)
    )
    env = CompileEnvironment.current()
    config = device_function.config

    def block_size(block_id: int) -> int:
        return env.size_hint(env.block_sizes[block_id].from_config_assert(config))

    mailboxes: dict[int, tuple[int, int, int, bool]] = {}
    words = 0
    for allocation_id in sorted(graph.inband_allocation_ids):
        # Sub-word elements share a tagged word when tile rows align to it.
        store = graph.inband_store(allocation_id)
        assert store.dtype is not None
        row = graph.inband_row(store, block_size)
        lanes = 4 // store.dtype.itemsize
        lanes = lanes if row % lanes == 0 else 1
        size = graph.inband_numel(allocation_id) // lanes
        pair = row % (2 * lanes) == 0 and size % 2 == 0
        mailboxes[allocation_id] = (words, size, lanes, pair)
        words += 2 * world_size * size
    mirrors: dict[int, TaggedMirror] = {}
    for allocation_id in sorted(graph.tagged_allocation_ids):
        # A mirror word holds lanes elements of one producer tile (T1, T2); a
        # mirror starts at a 16-byte boundary.
        store = graph.tagged_stores(allocation_id)[0]
        share = _pushed_share(graph, store, block_size, world_size)
        if share is None:
            raise TaggedPeerFallback(
                f"T7: {store.tensor_name}: each rank pushes a share of one dim"
            )
        assert store.dtype is not None
        row = graph.inband_row(store, block_size)
        lanes = 4 // store.dtype.itemsize
        lanes = lanes if row % lanes == 0 else 1
        size = (
            math.prod(
                _concrete_integer(extent, description="tagged shape")
                for extent in store.tensor_shape
            )
            // lanes
        )
        pair = row % (2 * lanes) == 0 and size % 2 == 0
        words += words % 2
        mirror = TaggedMirror(
            offset=words, words=size, lanes=lanes, pair=pair, share=share
        )
        # Row presence: of a masked push, or of one a guard may skip (T4).
        if store.has_explicit_mask or not graph.runs_at_root(store):
            chunk, rows, chunks = _presence_layout(graph, store, block_size, row, lanes)
            mirror = dataclasses.replace(
                mirror,
                words=size + size // (chunk * rows),
                presence=size,
                chunk=chunk,
                rows=rows,
                chunks=chunks,
            )
        mirror = dataclasses.replace(mirror, words=mirror.words + mirror.words % 2)
        mirrors[allocation_id] = mirror
        words += 2 * mirror.words
    # Two mirrors per tagged allocation, 8 bytes a word, in every rank's state:
    # tagged words within the inband pushes' budget.
    if 16 * sum(m.words for m in mirrors.values()) > _INBAND_PUSH_BYTES:
        raise TaggedPeerFallback("tagged mirrors past their budget")
    scatters, scatter_roots, flag = _scatter_layout(
        graph, block_size, world_size, words
    )
    workers = _worker_count(device_function)
    words = flag + (1 + world_size + workers if scatters else 0)
    words = -(-words // _PEER_SLOT_WORDS) * _PEER_SLOT_WORDS
    producers = sorted(
        {
            edge.producer_root
            for edge in graph.edges
            for dependency in edge.access_dependencies
            if graph.transport(dependency) == "peer_counter"
        }
    )
    slots = {root: words + i * _PEER_SLOT_WORDS for i, root in enumerate(producers)}
    done = words + len(producers) * _PEER_SLOT_WORDS
    state = _register_cross_loop_state(
        device_function,
        name_hint="tile_dependency_peer_state",
        numel=str(done + 1),
        dtype=torch.uint64,
        symmetric=True,
    )
    ptrs = device_function.triton_persistent_state_args[-1]
    rank = device_function.new_var("tile_dependency_peer_rank", dce=True)
    bases = tuple(
        device_function.new_var(f"tile_dependency_peer_base_{peer}", dce=True)
        for peer in range(world_size)
    )
    # The launcher appends this rank to the table of per-rank state pointers.
    device_function.preamble.extend(
        [
            statement_from_string(f"{rank} = tl.load({ptrs} + {world_size})"),
            *(
                statement_from_string(
                    f"{base} = tl.load({ptrs} + {peer}).to(tl.pointer_type(tl.uint64))"
                )
                for peer, base in enumerate(bases)
            ),
        ]
    )
    device_function.peer_state = PeerState(
        state=state,
        epoch=device_function.new_var("tile_dependency_peer_epoch", dce=False),
        ptrs=ptrs,
        rank=rank,
        bases=bases,
        mailboxes=mailboxes,
        slots=slots,
        done=done,
        scatters=scatters,
        scatter_roots=scatter_roots,
        mirrors=mirrors,
        flag=flag,
        started=flag + 1,
        votes=flag + 1 + world_size,
    )
    return device_function.peer_state


def _pushed_share(
    graph: TileDependencyGraph,
    store: TileAccess,
    block_size: Callable[[int], int],
    world_size: int,
) -> int | None:
    """Rule T7: the ranks' pushes cover the tensor once. This rank's tiles fill
    one share of one dimension, ``[share, share + 1) * size / world``, and all
    of the others (row masks aside: row presence words say which rows were
    pushed); returns the share, or None.  The verdict does not depend on the
    rank, so every rank picks the same transport; the launcher checks at the
    first launch that the ranks' shares differ (``_new_symmetric_state``)."""
    family = graph.task_families[store.root]
    shape = [
        _concrete_integer(extent, description="tagged shape")
        for extent in store.tensor_shape
    ]
    spans = store.subscript_dense_spans or (None,) * len(store.subscript_dims)
    bounds: list[tuple[int, int]] = []
    for dim, size in enumerate(shape):
        if dim not in store.subscript_dims:
            bounds.append((0, size))
            continue
        position = store.subscript_dims.index(dim)
        block_id = store.subscript_affine_block_ids[position]
        offset = store.subscript_offsets[position]
        if store.subscript_is_full_slice[position]:
            bounds.append((0, size))
            continue
        # Whole tiles of the axis: tile c covers [c, c + 1) * block * scale.
        span = spans[position]
        if span is None and block_id is not None and offset is not None:
            span = (
                (block_id, 1, offset)
                if store.subscript_index_scales[position] == 1
                and not store.subscript_is_scalar[position]
                else None
            )
        if span is not None:
            axis = family.axis(span[0])
            extent = axis.extent if axis is not None else None
            if not (
                axis is not None
                and axis.canonical_origin
                and isinstance(extent, int | sympy.Integer)
                and int(extent) % block_size(span[0]) == 0
            ):
                return None
            bounds.append((span[2], span[2] + int(extent) * span[1]))
            continue
        # A static index or vector, e.g. ``start + hl.arange(n)``.
        static = store.subscript_static_extents[position] or (
            1 if store.subscript_is_scalar[position] else None
        )
        if block_id is not None or offset is None or static is None:
            return None
        bounds.append((offset, offset + static))
    shares = [dim for dim, size in enumerate(shape) if bounds[dim] != (0, size)]
    if len(shares) != 1:
        return None
    (dim,) = shares
    (begin, end), share = bounds[dim], shape[dim] // world_size
    if shape[dim] % world_size or end - begin != share or begin % share:
        return None
    return begin // share


def _presence_layout(
    graph: TileDependencyGraph,
    store: TileAccess,
    block_size: Callable[[int], int],
    row: int,
    lanes: int,
) -> tuple[int, int, int]:
    """(words per tile row, rows per tile, tiles per row block) of a masked (or
    guarded) tagged push's row presence words. Its tile must be one row block
    (the whole second-to-last dimension, a power of two of at most 32 rows: one
    bit each) by aligned whole words."""
    # T2 made the tensor dense with a concrete shape.
    shape = [
        _concrete_integer(extent, description="tagged shape")
        for extent in store.tensor_shape
    ]
    rows = shape[-2] if len(shape) >= 2 else 0
    if not (
        len(shape) >= 2
        and 0 < rows <= 32
        and rows & (rows - 1) == 0
        and store.subscript_dims == tuple(range(len(shape)))
        and store.subscript_is_full_slice[-2]
        and all(
            graph.inband_extent(store, dim, block_size) == 1
            for dim in range(len(shape) - 2)
        )
        and row % lanes == 0
        and (shape[-1] // lanes) % (row // lanes) == 0
    ):
        raise TaggedPeerFallback(
            f"{store.tensor_name}: a masked or guarded push of one row block of "
            "whole words"
        )
    return row // lanes, rows, shape[-1] // row


def _worker_count(device_function: DeviceFunction) -> int:
    return CompileEnvironment.current().config_spec.num_sm * cast(
        "int", device_function.config.get("num_sm_multiplier", 1)
    )


def _scatter_layout(
    graph: TileDependencyGraph,
    block_size: Callable[[int], int],
    world_size: int,
    words: int,
) -> tuple[dict[int, Scatter], dict[int, ScatterRoot], int]:
    """Records, cover tables and done words of the inband scatters from ``words``,
    and the first word after them."""
    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    stores = {
        allocation_id: graph.inband_store(allocation_id)
        for allocation_id in sorted(graph.inband_allocation_ids)
        if graph.scatter_position(allocation_id) is not None
    }
    scatter_roots: dict[int, ScatterRoot] = {}
    for root in sorted({store.root for store in stores.values()}):
        family = graph.task_families[root]
        counts = {
            # The R1 scatter check admitted only integer extents.
            axis.block_id: -(
                -int(cast("sympy.Integer", axis.extent)) // block_size(axis.block_id)
            )
            for axis in family.axes
        }
        mine = [store for store in stores.values() if store.root == root]
        if len({store.graph_id for store in mine}) != 1:
            raise exc.CrossLoopSchedulingError(
                "because a root's inband scatter stores must share one block"
            )
        last = max(mine, key=lambda store: store.graph_node_index)
        scatter_roots[root] = ScatterRoot(
            axis_order=family.logical_axis_order,
            counts=counts,
            tasks=math.prod(counts.values()),
            done=0,
            last=last.access_id,
        )
    scatters: dict[int, Scatter] = {}
    for allocation_id, store in stores.items():
        position = graph.scatter_position(allocation_id)
        assert position is not None
        node = next(
            node
            for node in device_ir.graphs[store.graph_id].graph.nodes
            if store.access_id in node.meta.get(TILE_ACCESS_META, ())
        )
        index = node.args[1][position]
        assert isinstance(index, torch.fx.Node)
        rows = index.meta["val"]
        if rows.ndim > 1:
            raise exc.CrossLoopSchedulingError(
                "because an inband scatter's row index must be a vector"
            )
        lanes = 1
        if rows.ndim == 1:
            size = rows.size(0)
            block_id = None if isinstance(size, int) else env.get_block_id(size)
            lanes = block_size(block_id) if block_id is not None else int(size)
        shape = tuple(int(extent) for extent in store.tensor_shape)
        extents = []
        for dim in range(len(shape)):
            extent = (
                1 if dim == position else graph.inband_extent(store, dim, block_size)
            )
            if extent is None:
                raise exc.CrossLoopSchedulingError(
                    "because an inband scatter must write aligned row blocks"
                )
            extents.append(extent)
        counts = [-(-n // extent) for n, extent in zip(shape, extents, strict=True)]
        strides = [math.prod(counts[dim + 1 :]) for dim in range(len(shape))]
        tasks = scatter_roots[store.root].tasks
        scatters[allocation_id] = Scatter(
            root=store.root,
            position=position,
            lanes=lanes,
            shape=shape,
            extents=tuple(extents),
            key_strides=tuple(strides),
            keys=math.prod(counts),
            records=words,
            cover=words + 2 * tasks * lanes,
        )
        words += 2 * tasks * lanes + world_size * math.prod(counts)
    for root, layout in scatter_roots.items():
        scatter_roots[root] = dataclasses.replace(layout, done=words)
        words += 2 * world_size * layout.tasks
    return scatters, scatter_roots, words


def _outline_cross_loop_region(
    device_function: DeviceFunction,
    *,
    name_hint: str,
    body: list[ast.stmt],
    extra_argument_names: tuple[str, ...] = (),
    noinline: bool = False,
) -> ast.stmt:
    """Outline a scheduled region while keeping its computation opaque."""
    # Warp specialization (Triton's, or the ring kernel's partitions) changes
    # kernel-wide thread participation. Keep scheduler helpers inline so Triton
    # lowers their explicit and implicit barriers for the caller's active warp
    # group rather than the physical CTA.
    config = device_function.config
    helper_name, arguments = device_function.register_triton_outlined_helper(
        name_hint,
        body,
        extra_argument_names=extra_argument_names,
        noinline=noinline
        and not (
            any(CompileEnvironment.current().backend.range_warp_specializes(config))
            or device_function.ring is not None
        ),
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
        and (
            (node.func.value.id == "tl" and node.func.attr in {"dot", "dot_scaled"})
            # Gluon ring sites (``gluon_ring`` / ``gluon_ops`` in generated
            # code) carry the kernel's ring step and TMEM accumulators.
            or node.func.value.id in {"_helion_ring", "_helion_gluon"}
        )
        for statement in body
        for node in ast.walk(statement)
    )


_PURE_TL_CALLS = frozenset(
    {
        "arange",
        "associative_scan",
        "broadcast_to",
        "cast",
        "cdiv",
        "cumsum",
        "expand_dims",
        "full",
        "join",
        "load",
        "max",
        "maximum",
        "min",
        "minimum",
        "permute",
        "reduce",
        "reshape",
        "split",
        "sum",
        "where",
        "zeros",
    }
)
# Hoisting these from a persistent task loop saves memory traffic or a reduction.
_HOIST_ANCHOR_CALLS = frozenset(
    {"associative_scan", "cumsum", "load", "max", "min", "reduce", "sum"}
)


@dataclasses.dataclass(frozen=True)
class _GuardedExtent:
    """A root whose slowest-axis guard against a task-invariant bound became a
    dynamic task count, with the bound's slice hoisted out of the task loop."""

    hoisted: list[ast.stmt]
    body: list[ast.stmt]
    hoisted_names: tuple[str, ...]
    live_tasks: str


def _is_tl_call(
    node: ast.AST,
    names: frozenset[str] | None,
    modules: tuple[str, ...] = ("tl",),
) -> TypeGuard[ast.Call]:
    """Whether ``node`` calls ``module.name`` (any name for None)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in modules
        and (names is None or node.func.attr in names)
    )


def _is_load(node: ast.AST) -> TypeGuard[ast.Call]:
    """A pointer or descriptor load: ``tl.load``, ``desc.load``, or the backend's
    ``tl.load_tensor_descriptor`` (``Backend.descriptor_call``)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "load_tensor_descriptor")
    )


def _is_pure_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "to":
        return True
    return isinstance(func.value, ast.Name) and (
        func.value.id in ("libdevice", "tl_math")
        or (func.value.id == "tl" and func.attr in _PURE_TL_CALLS)
    )


def _guarded_extent(
    body: list[ast.stmt],
    *,
    slowest: PIDInfo,
    block_size: int,
    inner_tasks: int,
    variant_names: frozenset[str],
    read_only_tensors: frozenset[str],
    tensor_names: frozenset[str],
) -> _GuardedExtent | None:
    """Turn a trailing ``if offset < bound`` on the slowest PID axis into a task count.

    Pure prefix statements may move when their reaching definitions are invariant.
    Hoisting preserves dependencies between assignments, including repeated writes
    and compile-time assertions, without rewriting the original statements.
    """
    if not body or not isinstance(guard := body[-1], ast.If):
        return None
    prefix = body[:-1]
    if not isinstance(guard.test, ast.Name) or not all(
        isinstance(statement, ast.Pass) for statement in guard.orelse
    ):
        return None

    def target(statement: ast.stmt) -> str | None:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            node = statement.targets[0]
        elif isinstance(statement, ast.AugAssign):
            node = statement.target
        else:
            return None
        return node.id if isinstance(node, ast.Name) else None

    stored_names = _stored_names(body)
    body_writes = _stored_names(guard.body)
    targets: list[str | None] = []
    values: list[ast.expr] = []
    references: list[dict[str, int | None]] = []
    definitions: dict[str, list[int]] = {}
    reaching: dict[str, int] = {}
    invariant: set[int] = set()

    def is_invariant(value: ast.expr, reads: dict[str, int | None]) -> bool:
        for node in ast.walk(value):
            if isinstance(node, ast.Name):
                definition = reads[node.id]
                if (
                    node.id in variant_names
                    or (definition is None and node.id in stored_names)
                    or (definition is not None and definition not in invariant)
                ):
                    return False
            if (
                isinstance(node, ast.Call)
                and _is_tl_call(node, frozenset({"load"}))
                and (
                    not node.args
                    or any(keyword.arg == "volatile" for keyword in node.keywords)
                    or any(
                        pointer.id in tensor_names
                        and pointer.id not in read_only_tensors
                        for pointer in ast.walk(node.args[0])
                        if isinstance(pointer, ast.Name)
                    )
                )
            ):
                return False
        return True

    for index, statement in enumerate(prefix):
        name = target(statement)
        compile_time_assert = isinstance(statement, ast.Expr) and _is_tl_call(
            statement.value, frozenset({"static_assert"})
        )
        if not (
            (
                isinstance(statement, (ast.Assign, ast.AugAssign))
                and name is not None
                and (isinstance(statement, ast.Assign) or name in variant_names)
            )
            or compile_time_assert
        ):
            return None
        value = cast("ast.Assign | ast.AugAssign | ast.Expr", statement).value
        if any(
            (
                isinstance(node, ast.Call)
                and not (compile_time_assert and node is value)
                and not _is_pure_call(node)
            )
            or (isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load))
            for node in ast.walk(value)
        ):
            return None
        reads = {
            node.id: reaching.get(node.id)
            for node in ast.walk(value)
            if isinstance(node, ast.Name)
        }
        if isinstance(statement, ast.AugAssign):
            assert name is not None
            reads[name] = reaching.get(name)
        targets.append(name)
        values.append(value)
        references.append(reads)
        if not isinstance(statement, ast.AugAssign) and is_invariant(value, reads):
            invariant.add(index)
        if name is not None:
            definitions.setdefault(name, []).append(index)
            reaching[name] = index

    test = guard.test.id
    predicate_index = reaching.get(test)
    if predicate_index is None:
        return None
    predicate = values[predicate_index]
    if not isinstance(predicate, ast.Compare) or len(predicate.ops) != 1:
        return None
    left, right = predicate.left, predicate.comparators[0]
    if isinstance(predicate.ops[0], ast.Gt):
        bound, offset = left, right
    elif isinstance(predicate.ops[0], ast.Lt):
        bound, offset = right, left
    else:
        return None
    if not isinstance(bound, ast.Name) or not isinstance(offset, ast.Name):
        return None
    predicate_reads = references[predicate_index]
    bound_index = predicate_reads[bound.id]
    if (
        (bound_index is None and bound.id in stored_names | variant_names)
        or (bound_index is not None and bound_index not in invariant)
        or reaching.get(bound.id) != bound_index
    ):
        return None

    # Follow aliases to the exact tile-begin expression used by the predicate.
    offset_index = predicate_reads[offset.id]
    if offset_index is None:
        return None
    pid = slowest.pid_var
    offset_value = values[offset_index]
    while isinstance(offset_value, ast.Name) and offset_value.id != pid:
        alias = references[offset_index][offset_value.id]
        if alias is None:
            break
        offset_index = alias
        offset_value = values[offset_index]
    if ast.unparse(offset_value) not in (
        f"{pid} * {slowest.block_size_var}",
        *([pid] * (block_size == 1)),
    ):
        return None

    # Original names are retained, so all definitions of a moved name must move.
    hoistable = {
        index
        for name, writes in definitions.items()
        if name not in body_writes and all(write in invariant for write in writes)
        for index in writes
    } | {index for index in invariant if targets[index] is None}
    while True:
        invalid = {
            index
            for index in hoistable
            if any(
                definition is not None and definition not in hoistable
                for definition in references[index].values()
            )
        }
        if not invalid:
            break
        invalid.update(
            write
            for index in tuple(invalid)
            if (name := targets[index]) is not None
            for write in definitions[name]
        )
        hoistable.difference_update(invalid)
    if bound_index is not None and bound_index not in hoistable:
        return None

    pending = [
        index
        for index in sorted(hoistable)
        if targets[index] is None
        or index == bound_index
        or any(
            _is_tl_call(node, _HOIST_ANCHOR_CALLS) for node in ast.walk(values[index])
        )
    ]
    hoisted: set[int] = set()
    while True:
        while pending:
            index = pending.pop()
            if index in hoisted:
                continue
            if index not in hoistable:
                return None
            hoisted.add(index)
            pending.extend(
                definition
                for definition in references[index].values()
                if definition is not None
            )
            if (name := targets[index]) is not None:
                pending.extend(definitions[name])
        moved_definitions = {
            name: writes[-1]
            for name, writes in definitions.items()
            if writes[-1] in hoisted
        }
        # A statement that reads an earlier version must precede the later write.
        # Move it too when invariant; otherwise the original names cannot be kept.
        pending = [
            index
            for index, reads in enumerate(references)
            if index not in hoisted
            and any(
                name in moved_definitions and definition != moved_definitions[name]
                for name, definition in reads.items()
            )
        ]
        if not pending:
            break

    # Retain a predicate used by another prefix statement or by the guarded body.
    remove_predicate = not any(
        predicate_index in reads.values()
        for index, reads in enumerate(references)
        if index != predicate_index
    ) and test not in _all_names(guard.body)
    hoisted_statements = [
        _clone_stmt(statement)
        for index, statement in enumerate(prefix)
        if index in hoisted
    ]
    remaining = [
        _clone_stmt(statement)
        for index, statement in enumerate(prefix)
        if index not in hoisted and not (remove_predicate and index == predicate_index)
    ]
    tiles = bound.id if block_size == 1 else f"tl.cdiv({bound.id}, {block_size})"
    return _GuardedExtent(
        hoisted=hoisted_statements,
        body=[*remaining, *(_clone_stmt(statement) for statement in guard.body)],
        hoisted_names=tuple(
            dict.fromkeys(
                name
                for index, name in enumerate(targets)
                if index in hoisted and name is not None
            )
        ),
        live_tasks=tiles if inner_tasks == 1 else f"{inner_tasks} * {tiles}",
    )


class _Rename(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return create(ast.Name, id=self.names.get(node.id, node.id), ctx=node.ctx)


def _reuse_guard_values(body: list[ast.stmt], candidates: set[int]) -> None:
    """Value-number the kernel scope: each candidate ``name = value`` (a guard's
    hoisted statement every worker runs there) whose value a name still holds
    becomes an alias of that name, so roots guarded by the same bound reuse the
    first one's loads and reductions. A guard loads only tensors no later
    root writes (read-only or settled), so no statement invalidates a value."""
    numbers: dict[str, int] = {}  # name -> the number of the value it holds
    holders: dict[int, set[str]] = {}
    values: dict[str, int] = {}  # value -> its number
    unknown = 0

    def assign(name: str, number: int) -> None:
        holders.get(numbers.get(name, 0), set()).discard(name)
        numbers[name] = number
        holders.setdefault(number, set()).add(name)

    for index, statement in enumerate(body):
        if id(statement) not in candidates:
            for name in _stored_names([statement]):
                unknown += 1
                assign(name, unknown)
            continue
        assert isinstance(statement, ast.Assign)
        (target,) = (cast("ast.Name", t).id for t in statement.targets)
        renamed = _Rename({name: f"_v{number}" for name, number in numbers.items()})
        key = ast.dump(
            renamed.visit(cast("ast.AST", _clone_ast_value(statement.value)))
        )
        if key in values:
            number = values[key]
            holder = min(holders.get(number, set()) - {target}, default=None)
            if holder is not None:
                body[index] = statement_from_string(f"{target} = {holder}")
        else:
            number = values[key] = -1 - len(values)
        assign(target, number)


def _stored_names(nodes: Iterable[ast.AST]) -> set[str]:
    return {
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _all_names(nodes: Iterable[ast.AST]) -> set[str]:
    return {
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name)
    }


# Device modules a dry pass runs unchanged: memory goes through ``tl`` only.
_DRY_PASS_PURE_MODULES = frozenset(
    {"tl", "triton_helpers", "helion_triton_helpers", "tl_math", "libdevice"}
)
# Inductor helpers that only compute; the others load, spin, or publish.
_DRY_PASS_PURE_HELPERS = frozenset(
    {
        *("maximum", "minimum", "max2", "min2", "prod", "any"),
        *("promote_to_tensor", "div_floor_integer", "remainder_integer"),
        *("max_with_index", "min_with_index"),
        *("welford", "welford_reduce", "welford_combine"),
    }
)
# Helion's own Triton helpers that only reduce; ``triton_wait_signal`` spins.
_DRY_PASS_PURE_HELION_HELPERS = frozenset({"max_propagate_nan", "min_propagate_nan"})
# Tensor methods; descriptor methods such as ``load`` and ``store`` are skipped.
_DRY_PASS_PURE_METHODS = frozenset(
    {"to", "reshape", "view", "trans", "permute", "broadcast_to", "expand_dims"}
)


# Builtins Triton lowers to scalar arithmetic (PID decompositions call ``min``).
_PURE_BUILTINS = frozenset(
    ["min", "max", "abs", "int", "float", "bool", "len", "range"]
)


def _is_prefetch(call: ast.Call) -> bool:
    """Whether ``call`` is the inline asm of an ``hl.prefetch``."""
    asm = call.args[0] if call.args else None
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "inline_asm_elementwise"
        and isinstance(asm, ast.Constant)
        and asm.value in PREFETCH_ASMS
    )


def _effect_kind(call: ast.Call) -> str:
    """``_dry_pass_call_kind`` of a call outside a dry pass: pure builtins too,
    and prefetches, which order nothing (dropping one never changes a result)."""
    if isinstance(call.func, ast.Name) and call.func.id in _PURE_BUILTINS:
        return "pure"
    if _is_prefetch(call):
        return "pure"
    return _dry_pass_call_kind(call, frozenset())


def _dry_pass_call_kind(call: ast.Call, helpers: frozenset[str]) -> str:
    """Classify a call as ``helper``, ``masked``, ``skipped``, or ``pure``."""
    func = call.func
    if isinstance(func, ast.Name):
        return "helper" if func.id in helpers else "skipped"
    if not isinstance(func, ast.Attribute):
        return "skipped"
    if not (
        isinstance(func.value, ast.Name) and func.value.id in _DRY_PASS_PURE_MODULES
    ):
        return "pure" if func.attr in _DRY_PASS_PURE_METHODS else "skipped"
    if func.value.id == "triton_helpers":
        return "pure" if func.attr in _DRY_PASS_PURE_HELPERS else "skipped"
    if func.value.id == "helion_triton_helpers":
        return "pure" if func.attr in _DRY_PASS_PURE_HELION_HELPERS else "skipped"
    if func.value.id != "tl":
        return "pure"
    if func.attr == "inline_asm_elementwise":
        # Impure asm, such as the compiler's own polls and syncs, may spin or publish.
        pure = next((k.value for k in call.keywords if k.arg == "is_pure"), None)
        pure = call.args[4] if pure is None else pure
        if isinstance(pure, ast.Constant) and pure.value is True:
            return "pure"
        return "skipped"
    if func.attr in ("load", "store") or (
        func.attr.startswith("atomic_") and func.attr != "atomic_cas"
    ):
        # Block pointers take no mask.
        if any(k.arg == "boundary_check" for k in call.keywords):
            return "skipped"
        return "masked"
    if func.attr.startswith(
        ("load", "store", "atomic_", "make_block_ptr", "advance", "device_")
    ):
        return "skipped"
    return "pure"


def _dry_pass_mask(call: ast.Call, live: str) -> None:
    """AND the live predicate into a pointer load, store, or atomic mask."""
    load = cast("ast.Attribute", call.func).attr == "load"

    def update(slot: int, name: str, value: Callable[[ast.expr], ast.expr]) -> None:
        if len(call.args) > slot:
            call.args[slot] = value(call.args[slot])
            return
        keyword = next((k for k in call.keywords if k.arg == name), None)
        if keyword is not None:
            keyword.value = value(keyword.value)
        else:
            none = cast("ast.expr", expr_from_string("None"))
            call.keywords.append(create(ast.keyword, arg=name, value=value(none)))

    def masked(mask: ast.expr) -> ast.expr:
        if isinstance(mask, ast.Constant) and mask.value is None:
            return cast("ast.expr", expr_from_string(live))
        return cast("ast.expr", expr_from_string(f"({{mask}}) & {live}", mask=mask))

    def zero(other: ast.expr) -> ast.expr:
        if isinstance(other, ast.Constant) and other.value is None:
            return cast("ast.expr", expr_from_string("0.0"))
        return other

    update(1 if load else 2, "mask", masked)
    if load:
        # Masked lanes are undefined per thread; zeros keep dry branches uniform.
        update(2, "other", zero)


def _dry_pass_one_trip(loop: ast.expr, live: str) -> None:
    """Cut a ``tl.range`` loop to one trip when not ``live``: a single trip
    runs every instruction, which is all the code warm-up needs."""
    if not (
        isinstance(loop, ast.Call)
        and isinstance(loop.func, ast.Attribute)
        and isinstance(loop.func.value, ast.Name)
        and loop.func.value.id == "tl"
        and loop.func.attr == "range"
        and loop.args
    ):
        return
    # At most one trip, none where the real loop runs none.
    if len(loop.args) == 1:
        loop.args[0] = cast(
            "ast.expr",
            expr_from_string(
                f"tl.where({live}, {{stop}}, tl.minimum({{cap}}, 1))",
                stop=loop.args[0],
                cap=cast("ast.expr", _clone_ast_value(loop.args[0])),
            ),
        )
        return
    step = loop.args[2] if len(loop.args) > 2 else expr_from_string("1")
    loop.args[1] = cast(
        "ast.expr",
        expr_from_string(
            f"tl.where({live}, {{stop}}, tl.minimum({{cap}}, {{start}} + {{step}}))",
            stop=loop.args[1],
            cap=cast("ast.expr", _clone_ast_value(loop.args[1])),
            start=cast("ast.expr", _clone_ast_value(loop.args[0])),
            step=cast("ast.expr", _clone_ast_value(step)),
        ),
    )


def _dry_pass_rewrite(
    body: list[ast.stmt],
    live: str,
    helpers: frozenset[str],
    guards: list[ast.If],
) -> list[ast.stmt]:
    """Mask memory in ``body``; guard waits and unmaskable effects by ``live``."""
    result: list[ast.stmt] = []
    for statement in body:
        compound = isinstance(statement, (ast.If, ast.For))
        header = (
            (statement.test if isinstance(statement, ast.If) else statement.iter)
            if compound
            else statement
        )
        calls = [node for node in ast.walk(header) if isinstance(node, ast.Call)]
        kinds = [_dry_pass_call_kind(call, helpers) for call in calls]
        if isinstance(statement, ast.While) or "skipped" in kinds:
            # Helpers called under the guard still take the predicate.
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and _dry_pass_call_kind(node, helpers) == "helper"
                ):
                    node.args.append(cast("ast.expr", expr_from_string(live)))
            # Adjacent guarded statements share one guard so values flow.
            if guards and result and result[-1] is guards[-1]:
                guards[-1].body.append(statement)
            else:
                guards.append(
                    create(
                        ast.If, test=expr_from_string(live), body=[statement], orelse=[]
                    )
                )
                result.append(guards[-1])
            continue
        for call, kind in zip(calls, kinds, strict=True):
            if kind == "helper":
                call.args.append(cast("ast.expr", expr_from_string(live)))
            elif kind == "masked":
                _dry_pass_mask(call, live)
        if isinstance(statement, ast.For):
            _dry_pass_one_trip(statement.iter, live)
        if compound:
            statement.body = _dry_pass_rewrite(statement.body, live, helpers, guards)
            statement.orelse = _dry_pass_rewrite(
                statement.orelse, live, helpers, guards
            )
        result.append(statement)
    return result


def _dry_pass_binds(
    body: list[ast.stmt], guards: list[ast.If], bound: set[str]
) -> bool:
    """Whether every name a guard binds is read only under that guard."""
    guard_ids = {id(guard) for guard in guards}
    stores: dict[str, set[int]] = {name: {0} for name in bound}
    loads: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: int) -> None:
        scope = id(node) if id(node) in guard_ids else scope
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                stores.setdefault(node.id, set()).add(scope)
            else:
                loads.append((node.id, scope))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    for statement in body:
        visit(statement, 0)
    return all(name not in stores or stores[name] & {0, scope} for name, scope in loads)


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
    pipeline = device_function.config.cross_loop_pipeline
    dependency_graph = tagged_codegen_graph(device_function)
    assert dependency_graph is not None
    if dependency_graph.crosses_ranks() and pipeline == "barrier":
        raise AssertionError("cross-rank tile dependencies cannot use 'barrier'")
    if pipeline == "barrier":
        device_function.has_barrier = True
        return owner._emit_phase_loops(strategy, device_function, total_expr)
    if pipeline not in ("static", "dynamic"):
        raise exc.InvalidConfig(f"unknown cross_loop_pipeline value {pipeline!r}")
    # A ring kernel's persistent workers claim dynamic tasks root by root
    # (``claim_indices``); otherwise each dynamic program draws one ticket of a
    # launch-wide stream.
    dispatch_mode: CrossLoopDispatchMode = (
        "claimed"
        if pipeline == "dynamic" and device_function.ring is not None
        else cast("CrossLoopDispatchMode", pipeline)
    )
    # Guard-skipped tasks mark their inband scatter done words from each
    # worker's lanes, which static and claimed roots keep; packets have none (a
    # ring kernel regenerated without its ring runs static).
    if (
        dispatch_mode == "dynamic"
        and CompileEnvironment.current().config_spec.inband_scatter
    ):
        dispatch_mode = "static"

    configured_case_geometries = tuple(
        _static_case_geometry(owner, root, device_function)
        for root in range(len(owner.cases))
    )
    if any(geometry is None for geometry in configured_case_geometries):
        raise exc.InvalidConfig(
            f"cross_loop_pipeline={pipeline!r} requires representable "
            "top-level task counts"
        )
    case_geometries = tuple(
        geometry for geometry in configured_case_geometries if geometry is not None
    )
    worker = typed_program_id(0)
    epoch_var = device_function.new_var("tile_dependency_epoch", dce=False)
    case_bodies = cast("list[list[ast.stmt]]", owner.case_bodies)
    opaque_case_fingerprints = tuple(_ast_fingerprint(body) for body in case_bodies)
    target_device_capability = (
        CompileEnvironment.current().config_spec.target_device_capability
    )
    kernel_scope_roots = frozenset(
        root
        for root, body in enumerate(case_bodies)
        if _triton_root_requires_kernel_scope(body, target_device_capability)
    )
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
    configured_worker_count = _worker_count(device_function)
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
            f"cross_loop_pipeline={pipeline!r} requires representable root domains"
        )
    root_domains = tuple(
        domain for domain in configured_root_domains if domain is not None
    )
    root_task_orders = _root_task_orders(
        owner,
        root_domains,
        case_geometries,
    )
    if root_task_orders is None:
        raise exc.InvalidConfig(
            f"cross_loop_pipeline={pipeline!r} requires a fixed task capacity "
            "and representable root PID task order"
        )
    static_pipeline_plan = build_static_pipeline_plan(
        dependency_graph=dependency_graph,
        root_task_orders=root_task_orders,
        site_domains=site_domains,
        worker_count=configured_worker_count,
        publishable_site_ids=publishable_site_ids,
        # The ring's partitions replay task bodies, not continuations.
        continuation_ineligible_roots=(
            frozenset(range(len(root_domains)))
            if device_function.ring is not None
            else kernel_scope_roots
        ),
        prove_nonnegative=CompileEnvironment.current().known_nonnegative,
        cross_loop_dispatch_mode=dispatch_mode,
    )
    # StaticPipelinePlan accepts only a fixed physical task universe.  Convert
    # task-family offsets only after that invariant has been established.
    case_offsets: list[int] = []
    running_offset = 0
    for domain in root_domains:
        case_offsets.append(running_offset)
        running_offset += domain.size
    case_offset_strings = [str(offset) for offset in case_offsets]
    all_readiness_counter_plans = static_pipeline_plan.readiness_counters
    root_barrier_edges = static_pipeline_plan.root_barrier_edges
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
    launch_worker_count = static_pipeline_plan.worker_count
    root_barrier_producer_roots = sorted(
        {producer for producer, _consumer in root_barrier_edges}
    )
    if static_pipeline_plan.dispatch_mode != dispatch_mode:
        raise AssertionError("pipeline plan disagrees with configured dispatch mode")
    # Each root's lanes start where the earlier roots' tasks end, so short
    # later roots run on idle workers rather than behind every earlier root on
    # the first ones. Every wait the plan proves is wave-ordered, which any
    # per-root lane-to-worker bijection keeps; continuation ownership alone is
    # placed by slot residue (lane == worker), so such plans keep W-aligned lanes.
    packed_placement = not static_pipeline_plan.continuation_roots
    uses_packet_dispatch = dispatch_mode == "dynamic"
    packet_ranges: list[tuple[int, int, int]] = []
    packet_count = 0
    for root in static_pipeline_plan.resident_roots:
        packet_begin = packet_count
        packet_count += static_pipeline_plan.execution_orders[root].task_count
        packet_ranges.append((root, packet_begin, packet_count))
    if packet_count <= 0:
        raise AssertionError("dynamic packet stream requires at least one task")

    # Static ownership requires every worker to be simultaneously resident.
    # Dynamic packets rely on monotone admission and exact readiness instead.
    if not uses_packet_dispatch:
        device_function.triton_minimum_resident_programs = strategy.grid_size_expr
    device_function.preamble.extend(strategy._persistent_setup_statements(total_expr))
    if uses_packet_dispatch:
        strategy.grid_size_expr = str(packet_count)
    readiness_counter_offsets: dict[ReadinessCounterPlan, int] = {}
    readiness_counter_count = 0
    readiness_counter_stride = _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    for plan in all_readiness_counter_plans:
        readiness_counter_offsets[plan] = readiness_counter_count
        readiness_counter_count += (
            plan.readiness_key_domain.size * readiness_counter_stride
        )
    root_barrier_indices = {
        root: index for index, root in enumerate(root_barrier_producer_roots)
    }
    state_count = 0

    def reserve_state(count: int) -> int | None:
        nonlocal state_count
        if count == 0:
            return None
        state_count = (
            (state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
            // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
            * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        )
        offset = state_count
        state_count += count
        return offset

    root_barrier_count = (
        len(root_barrier_producer_roots) * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    # Claimed dispatch's multi-wave roots: the guarded ones among them (whose
    # live task counts depend on data) claim their tasks in order past a fixed
    # first wave (the ring's weight producer, or else a claimer partition,
    # claims and hands each claim to the others), so a worker that finishes
    # early takes more.
    claim_indices = {
        root: index
        for index, root in enumerate(
            root
            for root in static_pipeline_plan.resident_roots
            if dispatch_mode == "claimed"
            and static_pipeline_plan.execution_orders[root].task_count
            > launch_worker_count
        )
    }
    readiness_counter_state_offset = reserve_state(readiness_counter_count)
    root_barrier_state_offset = reserve_state(root_barrier_count)
    claim_state_offset = reserve_state(
        len(claim_indices) * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    epoch_state_count = launch_worker_count if not uses_packet_dispatch else 0
    counter_state_base = str(
        (epoch_state_count + _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS - 1)
        // _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
    )
    state_arg = (
        _register_cross_loop_state(
            device_function,
            name_hint="tile_dependency_state",
            numel=(f"{counter_state_base} + {state_count}"),
            dtype=_CROSS_LOOP_COUNTER_DTYPE,
        )
        if epoch_state_count or state_count
        else None
    )
    dispatch_ticket_arg = (
        _register_cross_loop_state(
            device_function,
            name_hint="tile_dependency_dispatch_ticket",
            numel="1",
            dtype=torch.uint64,
        )
        if uses_packet_dispatch
        else None
    )

    def state_section(offset: int | None) -> str | None:
        if offset is None:
            return None
        if state_arg is None:
            raise AssertionError("uint32 cross-loop state was not allocated")
        return f"{state_arg} + ({counter_state_base}) + {offset}"

    readiness_counter_arg = state_section(readiness_counter_state_offset)
    root_barrier_counter_arg = state_section(root_barrier_state_offset)
    claim_counter_arg = state_section(claim_state_offset)

    # peer_counter: one uint64 slot per publishing root plus a done slot, on
    # every rank. Targets use a uint64 epoch so they never wrap.
    peer_edges = static_pipeline_plan.peer_edges
    done_roots = static_pipeline_plan.done_roots
    peer = peer_state(device_function)
    world_size = len(peer.bases) if peer is not None else 1
    # A lost node key would turn a poll into a plain load of racing peer data.
    if any(
        dependency_graph.is_inband(access)
        and access.access_id not in device_function.inband_access_ids
        for access in dependency_graph.accesses
    ):
        raise AssertionError("an inband access was not emitted as a push or poll")

    def peer_target(roots: Iterable[int]) -> str:
        assert peer is not None
        arrivals = sum(static_pipeline_plan.publisher_count(root) for root in roots)
        return f"{peer.epoch} * {arrivals * world_size}"

    def peer_waits(root: int) -> list[ast.stmt]:
        producers = sorted(
            producer for producer, consumer in peer_edges if consumer == root
        )
        if not producers:
            return []
        assert peer is not None
        return [
            *(
                _bound_wait(
                    device_function,
                    f"helion_dist_utils._wait_at_least({peer.state} + "
                    f"{peer.slots[producer]}, {peer_target((producer,))})",
                )
                for producer in producers
            ),
            _publication_sync(device_function),
            *device_function.async_load_fence(),
        ]

    lanes = 1 << (world_size - 1).bit_length()
    threads = 32 * device_function.config.num_warps

    def peer_publications(root: int) -> list[ast.stmt]:
        if peer is None:
            return []
        slots = [peer.slots[root]] if any(p == root for p, _ in peer_edges) else []
        if root in done_roots:
            slots.append(peer.done)
        return [
            statement_from_string(
                f"helion_dist_utils._add_on_every_rank({peer.ptrs}, {slot}, "
                f"{world_size}, {lanes})"
            )
            for slot in slots
        ]

    # Keyed peer counters: one uint64 line per key on every rank, which each
    # producer task (live or skipped) bumps once per launch.
    peer_counter_plans = static_pipeline_plan.peer_counters
    peer_key_offsets: dict[ReadinessCounterPlan, int] = {}
    peer_key_words = 0
    for plan in peer_counter_plans:
        peer_key_offsets[plan] = peer_key_words
        peer_key_words += plan.readiness_key_domain.size * _PEER_SLOT_WORDS
    peer_keys = peer_key_ptrs = ""
    if peer_counter_plans:
        peer_keys = _register_cross_loop_state(
            device_function,
            name_hint="tile_dependency_peer_keys",
            numel=str(peer_key_words),
            dtype=torch.uint64,
            symmetric=True,
        )
        peer_key_ptrs = device_function.triton_persistent_state_args[-1]

    def peer_key_word(plan: ReadinessCounterPlan, key: str) -> str:
        return f"{peer_key_offsets[plan]} + ({key}) * {_PEER_SLOT_WORDS}"

    def peer_counter_publications(
        root: int, coordinates: dict[int, str], sem: str, guard: str | None = None
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for plan in peer_counter_plans:
            publication = plan.producers[0].incidence.keys_by_item
            if plan.producers[0].producer_root != root or publication is None:
                continue
            key, membership = relation_flat_target(
                publication,
                coordinates,
                trusted_single_valued=True,
                trusted_total_point_map=publication.is_total_function(),
            )
            # Scalar ptrs table only: outlined noinline tasks take no vectors.
            add = statement_from_string(
                f"helion_dist_utils._add_on_every_rank({peer_key_ptrs}, "
                f"{peer_key_word(plan, key)}, {world_size}, {lanes}, '{sem}')"
            )
            result.append(
                add
                if membership == "True"
                else create(
                    ast.If, test=expr_from_string(membership), body=[add], orelse=[]
                )
            )
        if guard is None or not result:
            return result
        return [create(ast.If, test=expr_from_string(guard), body=result, orelse=[])]

    def peer_counter_waits(
        root: int, coordinates: dict[int, str]
    ) -> tuple[list[ast.stmt], list[str]]:
        """Per-task acquire spins whose zero results gate the task's PID."""
        statements: list[ast.stmt] = []
        tokens: list[str] = []
        fence = (
            "fence.proxy.async.global; "
            if device_function.uses_async_proxy_global
            else ""
        )
        for plan in peer_counter_plans:
            arrivals = cast("int", plan.uniform_arrival_count()) * world_size
            for consumer in plan.consumers:
                if consumer.consumer_root != root:
                    continue
                assert peer is not None
                key, membership = relation_flat_target(
                    consumer.keys_by_consumer, coordinates
                )
                target = f"{peer.epoch} * {arrivals}"
                if membership != "True":
                    key = f"tl.where({membership}, {key}, 0)"
                    target = f"tl.where({membership}, {target}, 0)"
                token = device_function.new_var("tile_dependency_peer_token", dce=False)
                statements.append(
                    statement_from_string(
                        f"{token} = tl.inline_asm_elementwise("
                        "asm='{ .reg .pred p; .reg .b64 v; PEER_WAIT: "
                        "ld.acquire.sys.global.u64 v, [$1]; setp.lt.u64 p, v, $2; "
                        "@p bra PEER_WAIT; bar.warp.sync 0xffffffff; "
                        f"{fence}mov.u32 $0, 0; }}', constraints='=r,l,l', "
                        f"args=[{peer_keys} + {peer_key_word(plan, key)}, "
                        f"tl.cast({target}, tl.uint64)], dtype=tl.int32, "
                        "is_pure=False, pack=1)"
                    )
                )
                tokens.append(token)
                device_function.sync_roles[token] = SyncRole("wait", zero=True)
        return statements, tokens

    dispatch_ticket: str | None = None
    peer_launch = ""
    entry_gate: list[ast.stmt] = []
    if not uses_packet_dispatch:
        assert state_arg is not None
        result: list[ast.stmt] = [
            statement_from_string(f"{epoch_var} = tl.load({state_arg} + {worker}) + 1")
        ]
        device_function.sync_roles[epoch_var] = SyncRole("launch")
        if peer is not None:
            device_function.sync_roles[peer.epoch] = SyncRole("launch")
            # uint64 so peer targets never wrap.  Every worker counts its own
            # launches in its own word: each word has one reader and one writer
            # (that worker, which may read it from several partitions), so the
            # launch's end orders the rewrite after every read.
            launches = _register_cross_loop_state(
                device_function,
                name_hint="tile_dependency_peer_launch",
                numel=str(launch_worker_count),
                dtype=torch.uint64,
            )
            peer_launch = f"{launches} + {worker}"
            result.append(
                statement_from_string(f"{peer.epoch} = tl.load({peer_launch}) + 1")
            )
            if peer.scatters:
                # Read the hold flag now; its wait runs once the first loop's
                # prologue loads are in flight.
                held = device_function.new_var("tile_dependency_peer_held", dce=False)
                result.append(
                    statement_from_string(
                        f"{held} = tl.load({peer.state} + {peer.flag}, volatile=True)"
                        f" == {peer.epoch}"
                    )
                )
                # Announce this launch, then wait out last launch's reads of rows
                # no task wrote, which this launch may now overwrite.
                entry_gate = [
                    create(
                        ast.If,
                        test=expr_from_string(f"{worker} == 0"),
                        body=[
                            peer.every_rank_store(
                                device_function,
                                f"{peer.started} + {peer.rank}",
                                peer.epoch,
                                threads,
                            )
                        ],
                        orelse=[],
                    ),
                    _bound_wait(
                        device_function,
                        f"helion_dist_utils._hold_after_fallback({peer.state}, "
                        f"{peer.started}, {world_size}, {peer.epoch}, {held})",
                    ),
                ]
    else:
        assert dispatch_ticket_arg is not None
        raw_dispatch_ticket = device_function.new_var(
            "tile_dependency_raw_dispatch_ticket", dce=False
        )
        dispatch_ticket = device_function.new_var(
            "tile_dependency_dispatch_ticket", dce=True
        )
        period = f"tl.cast({packet_count}, tl.uint64)"
        result = [
            statement_from_string(
                f"{raw_dispatch_ticket} = tl.atomic_add("
                f"{dispatch_ticket_arg}, 1, sem='relaxed', scope='gpu')"
            ),
            statement_from_string(
                f"{dispatch_ticket} = tl.cast("
                f"{raw_dispatch_ticket} % {period}, tl.int32)"
            ),
            statement_from_string(
                f"{epoch_var} = tl.cast("
                f"{raw_dispatch_ticket} // {period} + 1, tl.uint32)"
            ),
        ]
        if peer is not None:
            result.append(
                statement_from_string(
                    f"{peer.epoch} = {raw_dispatch_ticket} // {period} + 1"
                )
            )
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
        arrivals = static_pipeline_plan.root_barrier_arrival_count(root)
        return (
            root_barrier_counter(root),
            f"tl.cast({epoch_var}, tl.uint32) * tl.cast({arrivals}, tl.uint32)",
        )

    def root_barrier_input_dependencies(
        root: int,
    ) -> tuple[tuple[str, str], ...]:
        producers = root_barrier_incoming.get(root, ())
        return tuple(root_barrier_dependency(producer) for producer in producers)

    # Per producer root: what each barrier wait every worker ran at kernel
    # scope orders (barrier_covers).  A launch's barrier stays complete, so a
    # later wait ordering no more is redundant.
    kernel_barrier_waits: dict[int, list[tuple[frozenset[int] | None, bool]]] = {}
    root_barrier_reachability = _root_barrier_reachability(root_barrier_edges)

    def barrier_covers(root: int, producer: int) -> tuple[frozenset[int] | None, bool]:
        """The allocations whose reads ``root``'s wait on ``producer``'s barrier
        orders (None: every access), and whether it also orders root's own
        barrier publication."""
        assert dependency_graph is not None
        # The barrier orders root's edges from producer and (barrier edges
        # being transitively reduced) from every root producer's barrier waits
        # for; one with an allocation unknown to the partitions orders every
        # access, as does one no edge explains.  In a root publishing a barrier
        # it also orders that publication: its waiters rely on it transitively
        # (without ring partitions, which keep waits by what they order, it
        # orders every access, so that later waits on producer are redundant).
        sources = {producer} | {
            source for source, target in root_barrier_reachability if target == producer
        }
        edges = [
            edge
            for edge in dependency_graph.edges
            if edge.producer_root in sources and edge.consumer_root == root
        ]
        covered = frozenset(
            access_allocations[dependency.consumer_access_id]
            for edge in edges
            for dependency in edge.access_dependencies
        )
        publication = root in root_barrier_indices
        if (
            not covered
            or any(edge.allocation_id < 0 for edge in edges)
            or not (covered <= set(argument_allocations.values()))
            or (publication and device_function.ring is None)
        ):
            return None, False
        return covered, publication

    def root_barrier_wait(
        root: int, producer: int, every_worker: bool
    ) -> list[ast.stmt]:
        """The wait on ``producer``'s barrier, recorded as ordering the accesses
        of ``root``'s edges from it (a partition making none skips it)."""
        covered, publication = barrier_covers(root, producer)
        earlier = kernel_barrier_waits.setdefault(producer, [])
        if any(
            orders is None
            or (
                covered is not None
                and covered <= orders
                and (published or not publication)
            )
            for orders, published in earlier
        ):
            # Every partition keeping this wait keeps the earlier one.
            return []
        if every_worker:
            earlier.append((covered, publication))
        counter, target = root_barrier_dependency(producer)
        wait = _wait_for_counter(
            device_function=device_function,
            counter=counter,
            target=target,
            prefix="tile_dependency_root_barrier_wait",
            load_fence=False,
        )
        if covered is None:
            return wait
        return _covering(device_function, wait, covered, publication)

    def root_barrier_arrivals(root: int, membership: str = "True") -> list[_Arrival]:
        if root not in root_barrier_indices:
            return []
        return [_Arrival(root_barrier_counter(root), membership=membership)]

    def root_barrier_publication(root: int, *, synced: bool = False) -> list[ast.stmt]:
        arrivals = root_barrier_arrivals(root)
        if not arrivals:
            return []
        return [
            *([] if synced else _release_sync(device_function)),
            *_emit_arrivals(device_function, list(arrivals), "release"),
        ]

    def folds_root_arrival(root: int) -> bool:
        """Whether a task's root-barrier arrival may ride its own release atomic.
        Not for kernel-scope roots: a task whose write guard fails arrives
        relaxed, which cannot certify the worker's earlier writes."""
        return root in root_barrier_indices and root not in kernel_scope_roots

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
    scheduled_task_roots = {
        root
        for root, order in enumerate(static_pipeline_plan.execution_orders)
        if order.ordinal_by_task
        != static_pipeline_plan.body_orders[root].ordinal_by_task
    }
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
    # A skipped task must not owe a per-task publication or reorder body PIDs.
    written_allocations = {
        access.allocation_id
        for access in dependency_graph.accesses
        if access.kind == "store"
    }
    access_names = {
        written: frozenset(
            access.tensor_name
            for access in dependency_graph.accesses
            if access.tensor_name is not None
            and (access.allocation_id in written_allocations) == written
        )
        for written in (False, True)
    }
    dependency_kinds = {
        dependency.dependency_id: dependency.kind
        for edge in dependency_graph.edges
        for dependency in edge.access_dependencies
    }
    # Roots whose writes supersede an earlier root's access to the same memory:
    # their consumers rely on their release for that earlier root's work too.
    superseding_roots = {
        edge.consumer_root
        for edge in dependency_graph.edges
        for dependency in edge.access_dependencies
        if dependency.kind is not TileDependencyKind.READ_AFTER_WRITE
    }
    # The allocation of each access (the device IR's view), and of each tensor
    # argument, for guard-first waits.
    access_allocations = _access_allocations()
    argument_allocations = _argument_allocations(device_function)
    dependencies_by_id = {
        dependency.dependency_id: dependency
        for edge in dependency_graph.edges
        for dependency in edge.access_dependencies
    }

    def consumer_covers(consumer: ReadinessConsumer) -> frozenset[int]:
        """The allocations of the reads whose obligations ``consumer``'s waits
        cover (what each of its waits orders)."""
        return frozenset(
            access_allocations[dependencies_by_id[obligation[0]].consumer_access_id]
            for obligation in consumer.covered_obligations
        )

    def arrives_unreleased(root: int) -> bool:
        """Whether a task of ``root`` that writes nothing may arrive without a
        release: not when its writes supersede an earlier root's access, a
        continuation runs on its arrival, or its counters cover WAR obligations
        (their consumers rely on its release)."""
        counters = root_counters_by_producer.get(root, ())
        return (
            root not in superseding_roots
            and all(plan.continuation_consumer is None for plan, _ in counters)
            and not any(
                dependency_kinds[obligation[0]] is TileDependencyKind.WRITE_AFTER_READ
                for plan, _producer in counters
                for consumer in plan.consumers
                for obligation in consumer.covered_obligations
            )
        )

    def warp_specialized(block_ids: list[int]) -> bool:
        specializes = CompileEnvironment.current().config_spec.range_warp_specialize
        return any(
            specializes.config_get(
                CompileEnvironment.current().backend.range_warp_specializes(
                    device_function.config
                ),
                block_id,
                None,
            )
            for block_id in block_ids
        )

    ws_roots = frozenset(
        root
        for root, case in enumerate(owner.cases)
        if warp_specialized([info.block_id for info in _case_pid_info(case)])
    )
    # Each worker's last keyed peer release of a root moves into the next
    # resident root's first task (StaticPipelinePlan.deferred_peer_releases):
    # the source root keeps the worker's last task id at kernel scope (-1 for
    # none) and the next root issues that release.
    deferred_releases = static_pipeline_plan.deferred_peer_releases(ws_roots)

    def invariant_wait(
        root: int, plan: ReadinessCounterPlan, consumer: ReadinessConsumer
    ) -> bool:
        """Whether ``root`` waits on ``plan`` once, before its task loop: the
        counter is the same for every task."""
        return (
            consumer.keys_by_consumer.is_total_function()
            and plan.readiness_key_domain.size == 1
            and not any(
                CompileEnvironment.current().backend.range_warp_specializes(
                    device_function.config
                )
            )
            and root not in deferred_releases
        )

    def settled_tensors(root: int) -> frozenset[str]:
        """Tensors ``root`` reads but never writes whose every in-kernel writer
        it awaits as a whole before its task loop (a root barrier, or a
        readiness wait whose counter is the same for every task): a load of
        one reads the same values in every task."""
        assert dependency_graph is not None
        invariant: set[int] = set()
        keyed: set[int] = set()
        for plan, consumer in readiness_consumers_by_root.get(root, ()):
            (invariant if invariant_wait(root, plan, consumer) else keyed).update(
                producer.producer_root for producer in plan.producers
            )
        barrier_producers = (
            set(root_barrier_incoming.get(root, ())) | invariant
        ) - keyed
        # Nor what a later root overwrites: a partition of a worker with no
        # live task may read the bound after its skipped arrivals.
        unsettled = (
            {
                access.allocation_id
                for access in dependency_graph.accesses
                if access.root == root and access.kind == "store"
            }
            | {
                edge.allocation_id
                for edge in dependency_graph.edges
                if edge.producer_root == root
            }
            | {
                edge.allocation_id
                for edge in dependency_graph.edges
                if edge.consumer_root == root
                and (
                    edge.producer_root not in barrier_producers
                    or any(
                        dependency_graph.transport(dependency) != "counter"
                        for dependency in edge.access_dependencies
                    )
                )
            }
        )
        return frozenset(
            access.tensor_name
            for access in dependency_graph.accesses
            if access.root == root
            and access.tensor_name is not None
            and access.allocation_id not in unsettled
        )

    guarded_extents: dict[int, _GuardedExtent] = {}
    # Extents whose hoisted loads read in-kernel results (``settled_tensors``).
    settled_extents: set[int] = set()
    for root in () if uses_packet_dispatch else static_pipeline_plan.resident_roots:
        axis_order, axis_counts, block_sizes = case_geometries[root]
        inner_counts = [axis_counts[block_id] for block_id in axis_order[:-1]]
        if (
            root in scheduled_task_roots
            # Skipped tasks arrive relaxed.
            or not arrives_unreleased(root)
            or root in nested_producer_roots
            or root in nested_loop_counters_by_consumer
            or isinstance(owner.cases[root], L2GroupingProgramIDs)
            or not all(isinstance(count, int) for count in inner_counts)
        ):
            continue
        settled = settled_tensors(root)
        extent = _guarded_extent(
            case_bodies[root],
            slowest=_case_pid_info(owner.cases[root])[-1],
            block_size=block_sizes[axis_order[-1]],
            inner_tasks=math.prod(cast("list[int]", inner_counts)),
            variant_names=frozenset({owner.shared_pid_var, strategy.virtual_pid_var}),
            read_only_tensors=access_names[False] - access_names[True] | settled,
            tensor_names=frozenset(
                argument.name
                for argument in device_function.arguments
                if isinstance(argument, TensorArg)
            ),
        )
        if extent is not None:
            guarded_extents[root] = extent
            if _all_names(extent.hoisted) & settled & access_names[True]:
                settled_extents.add(root)
    if tagged := dependency_graph.tagged_allocation_ids:
        failure = tagged_coverage_failure(dependency_graph, tagged, guarded_extents)
        if failure is not None:
            raise TaggedPeerFallback(failure)
    # Consumers wait on one done word per scatter task (or skip mark) per launch.
    for root in {} if peer is None else peer.scatter_roots:
        extent = guarded_extents.get(root)
        done = device_function.inband_scatter_done.get(root)
        if done is None or done not in _stored_names(
            statement
            for statement in (case_bodies[root] if extent is None else extent.body)
            if isinstance(statement, ast.Assign)
        ):
            raise exc.CrossLoopSchedulingError(
                "because an inband scatter store must run once in every task"
            )

    # Roots that poll in-band peer data wait on it in their bodies.
    inband_consumer_roots = {
        access.root
        for access in dependency_graph.accesses
        if dependency_graph.is_inband(access) and access.kind != "store"
    }

    def waits_on_dependency(root: int) -> bool:
        return bool(
            root_barrier_incoming.get(root)
            or readiness_consumers_by_root.get(root)
            or nested_loop_counters_by_consumer.get(root)
            or any(consumer == root for _producer, consumer in peer_edges)
            or root in inband_consumer_roots
        )

    def dry_pass_task(task_body: list[ast.stmt]) -> list[ast.stmt] | None:
        """Run a waiting task once with memory masked and waits skipped.

        The dry pass warms the task's code while it would otherwise wait. Both
        passes share one loop body; None if a skipped value would be read.
        """
        helpers = {
            name: (arguments, body)
            for name, arguments, body, _noinline in (
                device_function.triton_outlined_helpers
            )
        }
        reachable: list[str] = []
        pending = [task_body]
        while pending:
            for node in (n for s in pending.pop() for n in ast.walk(s)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in helpers
                    and node.func.id not in reachable
                ):
                    reachable.append(node.func.id)
                    pending.append(list(helpers[node.func.id][1]))
        live = device_function.new_var("tile_dependency_live", dce=False)

        def rewrite(body: Iterable[ast.stmt], bound: set[str]) -> list[ast.stmt] | None:
            guards: list[ast.If] = []
            rewritten = _dry_pass_rewrite(
                [_clone_stmt(statement) for statement in body],
                live,
                frozenset(reachable),
                guards,
            )
            return rewritten if _dry_pass_binds(rewritten, guards, bound) else None

        helper_bodies = {
            name: rewrite(helpers[name][1], set(helpers[name][0])) for name in reachable
        }
        branch_body = rewrite(task_body, set())
        if branch_body is None or None in helper_bodies.values():
            return None
        outlined = device_function.triton_outlined_helpers
        for index, (name, arguments, _body, noinline) in enumerate(outlined):
            if (helper_body := helper_bodies.get(name)) is not None:
                outlined[index] = (name, (*arguments, live), (*helper_body,), noinline)
        dry_pass = device_function.new_var("tile_dependency_dry_pass", dce=False)
        # A runtime zero start keeps ptxas from unrolling the two passes.
        return [
            create(
                ast.For,
                target=create(ast.Name, id=dry_pass, ctx=ast.Store()),
                iter=expr_from_string(
                    f"tl.range({dispatch_ticket} // {1 << 30}, 2, 1, num_stages=1)"
                ),
                body=[
                    statement_from_string(f"{live} = {dry_pass} == 1"),
                    *branch_body,
                ],
                orelse=[],
                type_comment=None,
            ),
        ]

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
                coordinates[block_id] = f"(({task}) % ({count_text}))"
            else:
                coordinates[block_id] = (
                    f"((({task}) // ({multiplier_text})) % ({count_text}))"
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
        *,
        nonempty_domain: CoordinateDomain | None = None,
    ) -> str:
        """Render the restricted coordinate-relation expression grammar."""

        def render(child: sympy.Expr) -> str:
            return relation_expression(
                child,
                coordinates,
                nonempty_domain=nonempty_domain,
            )

        def positive_when_domain_nonempty(divisor: sympy.Expr) -> bool:
            if divisor.is_positive is True:  # pyrefly: ignore[missing-attribute]
                return True
            if (
                nonempty_domain is None
                or divisor.is_nonnegative is not True  # pyrefly: ignore[missing-attribute]
                or not divisor.free_symbols <= nonempty_domain.parameter_symbols
            ):
                return False
            domain_size = sympy.sympify(nonempty_domain.size_expr)
            quotient = sympy.cancel(domain_size / divisor)
            return (
                quotient.is_integer is True
                and quotient.is_nonnegative is True
                and sympy.simplify(domain_size - divisor * quotient) == 0
            )

        def floor_division(
            numerator: sympy.Expr,
            denominator: sympy.Expr,
        ) -> str:
            if denominator.is_integer is not True or not (  # pyrefly: ignore[missing-attribute]
                positive_when_domain_nonempty(denominator)
            ):
                raise AssertionError(
                    "logical floor division requires a proved positive divisor"
                )
            numerator_text = render(numerator)
            if isinstance(denominator, sympy.Integer):
                divisor = str(int(denominator))
            else:
                # ``nonempty_domain`` proves this factor is positive at every
                # executed point. Clamping only defines the inactive empty-
                # domain case, which the surrounding schedule never executes.
                denominator_text = render(denominator)
                divisor = f"tl.maximum(({denominator_text}), 1)"
            if numerator.is_nonnegative is True:  # pyrefly: ignore[missing-attribute]
                return f"(({numerator_text}) // {divisor})"
            # Triton integer division truncates toward zero. First remove the
            # Euclidean remainder so the dividend is exactly divisible; this
            # preserves mathematical floor for either sign.
            signed_remainder = f"(({numerator_text}) % ({divisor}))"
            remainder = f"((({signed_remainder}) + {divisor}) % {divisor})"
            return f"((({numerator_text}) - ({remainder})) // {divisor})"

        if isinstance(expression, sympy.Integer):
            return str(int(expression))
        if expression.func in (sympy.Min, sympy.Max, SymbolicMin, SymbolicMax):
            function = (
                "tl.minimum"
                if expression.func in (sympy.Min, SymbolicMin)
                else "tl.maximum"
            )
            rendered = render(cast("sympy.Expr", expression.args[0]))
            for argument in expression.args[1:]:
                rendered = (
                    f"{function}(({rendered}), "
                    f"({render(cast('sympy.Expr', argument))}))"
                )
            return rendered
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
                f"({render(cast('sympy.Expr', term))})"
                for term in expression.as_ordered_terms()
            )
        if isinstance(expression, sympy.Mul):
            return " * ".join(
                f"({render(factor)})" for factor in expression.as_ordered_factors()
            )
        if expression.func in (FloorDiv, CeilDiv):
            numerator, denominator = map(sympy.sympify, expression.args)
            if expression.func is FloorDiv:
                return floor_division(numerator, denominator)
            return f"(-({floor_division(-numerator, denominator)}))"
        if expression.func in (sympy.floor, sympy.ceiling):
            numerator, denominator = sympy.fraction(sympy.together(expression.args[0]))
            if expression.func == sympy.floor:
                return floor_division(numerator, denominator)
            return f"(-({floor_division(-numerator, denominator)}))"
        if isinstance(expression, sympy.Mod):
            numerator, denominator = expression.args
            if not isinstance(denominator, sympy.Integer) or denominator <= 0:
                raise AssertionError(
                    "logical modulo requires a positive static divisor"
                )
            numerator_text = render(cast("sympy.Expr", numerator))
            signed_remainder = f"(({numerator_text}) % {int(denominator)})"
            if numerator.is_nonnegative is True:  # pyrefly: ignore[missing-attribute]
                return signed_remainder
            # SymPy's Mod is Euclidean for a positive divisor, whereas Triton
            # lowers ``%`` on signed integers as a signed remainder.  Preserve
            # the relation's semantics when affine simplification has removed
            # a positive multiple of the divisor from a potentially negative
            # numerator (for example, a wrapped worker cohort).
            return f"((({signed_remainder}) + {int(denominator)}) % {int(denominator)})"
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
        *,
        trusted_single_valued: bool = False,
        trusted_total_point_map: bool = False,
    ) -> tuple[dict[int, str], str]:
        """Render an at-most-one-valued relation without task tables."""
        # A finalized worker schedule has already proved every segment to be
        # an exact at-most-one-valued map.  Preserve that proof boundary:
        # canonicalizing its symbolic packed boxes again is both redundant and
        # can make code generation scale with nested Min/Max expression size.
        if trusted_single_valued or trusted_total_point_map:
            canonical = relation
        else:
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
                unit_width = step == 1 and (
                    sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                    == 1
                )
                if not unit_width and not (
                    trusted_single_valued or trusted_total_point_map
                ):
                    raise AssertionError("event relation target is not one point")
                value = relation_expression(
                    begin,
                    source_coordinates,
                    nonempty_domain=(
                        relation.target_domain if trusted_total_point_map else None
                    ),
                )
                piece_values[axis] = value
                if not unit_width and not trusted_total_point_map:
                    target_memberships.append(
                        f"({value}) < ({relation_expression(end, source_coordinates)})"
                    )
                target_count = relation_expression(
                    sympy.sympify(canonical.target_domain.axis_count_expressions[axis]),
                    source_coordinates,
                    nonempty_domain=(
                        relation.target_domain if trusted_total_point_map else None
                    ),
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
                if trusted_total_point_map
                or (not trusted_single_valued and canonical.is_total_function())
                else " or ".join(f"({membership})" for membership in memberships)
            ),
        )

    def relation_flat_target(
        relation: CoordinateRelation,
        source_coordinates: dict[int, str],
        *,
        trusted_single_valued: bool = False,
        trusted_total_point_map: bool = False,
    ) -> tuple[str, str]:
        target_coordinates, membership = relation_point_coordinates(
            relation,
            source_coordinates,
            trusted_single_valued=trusted_single_valued,
            trusted_total_point_map=trusted_total_point_map,
        )
        return (
            flat_task_from_coordinates(
                target_coordinates,
                relation.target_domain.axis_order,
                relation.target_domain.axis_count_expressions,
            ),
            membership,
        )

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
        placement = nested_wait_placement(
            root_domains[readiness_consumer.consumer_root], readiness_consumer
        )
        if placement is None:
            raise AssertionError(
                "nested loop lowering currently requires one loop axis"
            )
        nested_axis, segment_begin_iterations = placement
        covered = consumer_covers(readiness_consumer)
        if segment_begin_iterations is None:
            scheduled = False

            def rewrite(loop: ast.For) -> list[ast.stmt] | None:
                nonlocal scheduled
                if tile_dependency_site_id(loop) != readiness_consumer.consumer_site_id:
                    return None
                if scheduled:
                    raise AssertionError(
                        "one dependency site must identify one lowered loop"
                    )
                if (
                    not isinstance(loop.target, ast.Name)
                    or not isinstance(loop.iter, ast.Call)
                    or len(loop.iter.args) < 2
                ):
                    raise AssertionError(
                        "nested loop wait requires one range-like loop axis"
                    )
                begin = ast.unparse(loop.iter.args[0])
                step = (
                    ast.unparse(loop.iter.args[2]) if len(loop.iter.args) >= 3 else "1"
                )
                site_coordinates = {
                    **consumer_coordinates,
                    nested_axis: f"(({loop.target.id}) - ({begin})) // ({step})",
                }
                readiness_key, membership = relation_flat_target(
                    readiness_consumer.keys_by_consumer,
                    site_coordinates,
                )
                waits = _covering(
                    device_function,
                    _wait_for_counter(
                        device_function=device_function,
                        counter=readiness_counter(plan, readiness_key),
                        target=readiness_target(plan, readiness_key),
                        prefix="tile_dependency_nested_loop_wait",
                    ),
                    covered,
                )
                cloned = cast("ast.For", _clone_ast_value(loop))
                if membership == "True":
                    cloned.body = [*waits, *cloned.body]
                elif membership != "False":
                    cloned.body = [
                        create(
                            ast.If,
                            test=expr_from_string(membership),
                            body=waits,
                            orelse=[],
                        ),
                        *cloned.body,
                    ]
                scheduled = True
                return [cloned]

            rewritten = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
            if not scheduled:
                raise AssertionError(
                    f"missing dependency site {readiness_consumer.consumer_site_id}"
                )
            return rewritten

        boundaries = segment_begin_iterations[1:]
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
                    _covering(
                        device_function,
                        _wait_for_counter(
                            device_function=device_function,
                            counter=readiness_counter(plan, readiness_key),
                            target=readiness_target(plan, readiness_key),
                            prefix="tile_dependency_nested_loop_wait",
                        ),
                        covered,
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
        return (
            f"{readiness_counter_arg} + {offset} + "
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
            plan.readiness_key_domain.axis_count_expressions,
        )
        expressions: list[str] = []
        for readiness_producer in plan.producers:
            cardinality = readiness_producer.incidence.count_by_key
            if cardinality is None:
                raise AssertionError("event fan-in is not symbolically known")
            values, _membership = relation_point_coordinates(
                cardinality,
                readiness_key_coordinates,
            )
            expressions.append(values[cardinality.target_domain.axis_order[0]])
        return " + ".join(f"({expression})" for expression in expressions)

    def readiness_target(
        plan: ReadinessCounterPlan,
        readiness_key: str,
    ) -> str:
        arrivals = readiness_expected_arrivals(plan, readiness_key)
        return f"tl.cast({epoch_var}, tl.uint32) * tl.cast({arrivals}, tl.uint32)"

    def emit_readiness_arrival_for_key(
        plan: ReadinessCounterPlan,
        readiness_key: str,
    ) -> list[ast.Assign | _Arrival | _Inline]:
        continuation_consumer = plan.continuation_consumer
        if continuation_consumer is None:
            # Every counter takes one +1 per arrival, so the arrivals of a
            # publication point share one atomic (a counter of one arrival
            # per launch reads the epoch just like an exchange would).
            return [_Arrival(readiness_counter(plan, readiness_key))]

        consumers_by_key = continuation_consumer.incidence.items_by_key
        if not consumers_by_key.is_total_function():
            raise AssertionError(
                "a continuation event must bijectively cover its consumer"
            )
        if consumers_by_key.is_positional_bijection():
            consumer_task_expression = readiness_key
        else:
            readiness_key_coordinates = flat_task_coordinates(
                readiness_key,
                plan.readiness_key_domain.axis_order,
                plan.readiness_key_domain.axis_count_expressions,
            )
            consumer_coordinates, _membership = relation_point_coordinates(
                consumers_by_key,
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
            cast(
                "ast.Assign",
                statement_from_string(f"{consumer_task} = {consumer_task_expression}"),
            )
        ]
        continuation_root = continuation_consumer.consumer_root
        consumer_coordinates = flat_task_coordinates(
            consumer_task,
            root_domains[continuation_root].axis_order,
            root_axis_counts[continuation_root],
        )
        consumer_body_pid, body_pid_membership = relation_flat_target(
            static_pipeline_plan.body_orders[continuation_root].ordinal_by_task,
            consumer_coordinates,
            trusted_total_point_map=True,
        )
        if body_pid_membership != "True":
            raise AssertionError("continuation body PID ABI is not total")
        consumer_logical_pid = (
            f"{case_offset_strings[continuation_root]} + {consumer_body_pid}"
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
        consumer_parts: list[ast.Assign | _Arrival | _Inline] = list(
            root_barrier_arrivals(continuation_root)
        )
        for nested_counter, nested_producer in root_counters_by_producer.get(
            continuation_root, ()
        ):
            consumer_parts.extend(
                emit_readiness_arrivals_from_producer(
                    nested_counter,
                    nested_producer,
                    consumer_coordinates,
                )
            )
        consumer_publications = _emit_arrivals(
            device_function, consumer_parts, "release"
        )

        last_arrival_body = [*device_function.async_load_fence(), consumer_call]
        if consumer_publications:
            last_arrival_body.extend(_release_sync(device_function))
            last_arrival_body.extend(consumer_publications)
        expected_arrivals = plan.uniform_arrival_count()
        if expected_arrivals is None:
            raise AssertionError(
                "final-arrival continuation requires uniform readiness fan-in"
            )
        if expected_arrivals == 1:
            return [*assignments, _Inline(tuple(last_arrival_body))]
        return [
            *assignments,
            _Arrival(
                readiness_counter(plan, readiness_key),
                target=readiness_target(plan, readiness_key),
                previous=previous,
                body=tuple(last_arrival_body),
            ),
        ]

    def emit_readiness_arrivals_from_producer(
        plan: ReadinessCounterPlan,
        readiness_producer: ReadinessProducer,
        producer_coordinates: dict[int, str],
    ) -> list[ast.Assign | _Arrival | _Inline]:
        publication = readiness_producer.incidence.keys_by_item
        if publication is None:
            raise AssertionError("readiness publication relation is unavailable")
        readiness_key, membership = relation_flat_target(
            publication,
            producer_coordinates,
            trusted_single_valued=True,
        )
        return _arrival_parts_under(
            emit_readiness_arrival_for_key(plan, readiness_key), membership
        )

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
                readiness_producer.incidence.items_by_key.target_domain
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

            parts: list[ast.Assign | _Arrival | _Inline] = []
            for plan, readiness_producer in producer_counters_by_site[site_id]:
                publication = readiness_producer.incidence.keys_by_item
                if publication is None:
                    raise AssertionError(
                        "nested-loop readiness publication is unavailable"
                    )
                readiness_key, membership = relation_flat_target(
                    publication,
                    site_coordinates,
                    trusted_single_valued=True,
                )
                parts.extend(
                    _arrival_parts_under(
                        emit_readiness_arrival_for_key(plan, readiness_key),
                        membership,
                    )
                )
            publications = _emit_arrivals(device_function, parts, "release")

            cloned = cast("ast.For", _clone_ast_value(loop))
            cloned.body.extend([*_release_sync(device_function), *publications])
            emitted_site_ids.add(site_id)
            return [cloned]

        result = _clone_opaque_statements_with_loop_rewrite(body, rewrite)
        if emitted_site_ids != site_ids:
            missing = sorted(site_ids - emitted_site_ids)
            raise AssertionError(f"missing nested producer sites {missing}")
        return result

    def execution_coordinates(root: int, local_task: str) -> dict[int, str]:
        execution = static_pipeline_plan.execution_orders[root].tasks_by_ordinal
        coordinates, membership = relation_point_coordinates(
            execution,
            flat_task_coordinates(
                local_task,
                execution.source_domain.axis_order,
                execution.source_domain.axis_count_expressions,
            ),
            trusted_total_point_map=True,
        )
        if membership != "True":
            raise AssertionError("proved root execution order is not total")
        return coordinates

    def guarded_kernel_scope_root(root: int) -> bool:
        """A kernel-scope root with a write guard, which must stay first in its
        task body (its failed tasks arrive relaxed): its waits stay at task
        entry rather than being scoped."""
        extent = guarded_extents.get(root)
        body = case_bodies[root] if extent is None else extent.body
        return root in kernel_scope_roots and _write_guard(body) is not None

    def scheduled_root_task_body(
        root: int,
        root_local_pid_task: str,
        logical_pid: str,
        extra_argument_names: tuple[str, ...],
        *,
        force_noinline: bool = False,
        invariant_waits: list[ast.stmt] | None = None,
        barrier_waits: Sequence[tuple[list[ast.stmt], frozenset[int]]] = (),
        root_arrival: _Arrival | None = None,
    ) -> list[ast.stmt]:
        """One task of ``root``; waits whose counter is the same for every task
        go to ``invariant_waits`` when given (the caller waits once).
        ``barrier_waits``: root-barrier waits and the allocations they cover,
        placed like the covered readiness waits.  ``root_arrival``: the
        worker's root-barrier arrival, made with its last task's arrivals."""
        body: list[ast.stmt] = []
        extent = guarded_extents.get(root)
        root_body = case_bodies[root] if extent is None else extent.body
        has_task_scheduling = root in nested_loop_counters_by_consumer
        producer_counters = tuple(root_counters_by_producer.get(root, ()))
        scheduled_logical_pid = logical_pid
        scheduled_coordinates = execution_coordinates(root, root_local_pid_task)
        if producer_counters or root in nested_producer_roots:
            has_task_scheduling = True
        if root in scheduled_task_roots:
            pid_task = device_function.new_var(
                "tile_dependency_scheduled_pid_task", dce=True
            )
            pid_task_expression, pid_membership = relation_flat_target(
                static_pipeline_plan.body_orders[root].ordinal_by_task,
                scheduled_coordinates,
                trusted_total_point_map=True,
            )
            if pid_membership != "True":
                raise AssertionError("body PID ABI is not total")
            body.append(statement_from_string(f"{pid_task} = {pid_task_expression}"))
            scheduled_logical_pid = f"{case_offset_strings[root]} + {pid_task}"
        readiness_tokens: list[str] = []
        token_waits = any(
            CompileEnvironment.current().backend.range_warp_specializes(
                device_function.config
            )
        )
        # A task whose write guard fails touches nothing these waits protect,
        # so they can follow the guard, unless a consumer relies on the task's
        # arrival implying them (its writes supersede an earlier root's access,
        # or a continuation runs on the last arrival) or it goes to other ranks.
        guard_first = (
            root in kernel_scope_roots
            and not token_waits
            and root not in superseding_roots
            and not peer_publications(root)
            and not any(
                producer.producer_root == root
                for plan in peer_counter_plans
                for producer in plan.producers
            )
            and not any(
                plan.continuation_consumer is not None
                for plan, _consumer in (
                    *readiness_consumers_by_root.get(root, ()),
                    *root_counters_by_producer.get(root, ()),
                )
            )
        )
        guard_first_waits: list[tuple[list[ast.stmt], frozenset[int]]] = []
        # Waits that order only covered tensors move to the task's first
        # access of them, so its independent loads issue before the spin.
        scoped_waits: list[tuple[list[ast.stmt], frozenset[int]]] = []
        guard_first_index = len(body)
        guard_first_allocations: set[int] = set()
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
            counter = readiness_counter(incoming_readiness_counter, readiness_key)
            target = readiness_target(incoming_readiness_counter, readiness_key)
            total = incoming_consumer.keys_by_consumer.is_total_function()
            covered = consumer_covers(incoming_consumer)
            if invariant_waits is not None and invariant_wait(
                root, incoming_readiness_counter, incoming_consumer
            ):
                invariant_waits.extend(
                    _covering(
                        device_function,
                        _wait_for_counter(
                            device_function=device_function,
                            counter=counter,
                            target=target,
                            prefix="tile_dependency_readiness_wait",
                        ),
                        covered,
                    )
                )
                continue
            if token_waits:
                # Triton's warp specialization rejects a while loop in a task
                # loop: spin in one asm block and gate the task PID on its token.
                token = device_function.new_var(
                    "tile_dependency_readiness_token", dce=False
                )
                fence = (
                    "fence.proxy.async.global; "
                    if device_function.uses_async_proxy_global
                    else ""
                )
                wait: list[ast.stmt] = [
                    statement_from_string(
                        f"{token} = tl.inline_asm_elementwise("
                        "asm='{ .reg .pred p; .reg .b32 v; READY_WAIT: "
                        "ld.acquire.gpu.global.u32 v, [$1]; setp.ne.u32 p, v, $2; "
                        "@p bra READY_WAIT; bar.warp.sync 0xffffffff; "
                        f"{fence}mov.u32 $0, 0; }}', constraints='=r,l,r', "
                        f"args=[{counter}, tl.cast({target}, tl.uint32)], "
                        "dtype=tl.int32, is_pure=False, pack=1)"
                    )
                ]
                readiness_tokens.append(token)
                if not total:
                    body.append(statement_from_string(f"{token} = 0"))
            else:
                wait = _covering(
                    device_function,
                    _wait_for_counter(
                        device_function=device_function,
                        counter=counter,
                        target=target,
                        prefix="tile_dependency_readiness_wait",
                    ),
                    covered,
                )
            if not total:
                wait = [
                    create(
                        ast.If,
                        test=expr_from_string(membership),
                        body=wait,
                        orelse=[],
                    )
                ]
            if guard_first:
                guard_first_waits.append((wait, covered))
                guard_first_allocations.update(covered)
            elif covered and not token_waits:
                scoped_waits.append((wait, covered))
            else:
                body.extend(wait)
        peer_counter_spins, peer_tokens = peer_counter_waits(
            root, scheduled_coordinates
        )
        if readiness_tokens:
            scheduled_logical_pid = " + ".join(
                [scheduled_logical_pid, *readiness_tokens]
            )
        if peer_tokens:
            # Zero tokens make every PID-derived address depend on the spins.
            has_task_scheduling = True
            body.extend(peer_counter_spins)
            scheduled_logical_pid = " + ".join([scheduled_logical_pid, *peer_tokens])
        nested_loop_consumers = nested_loop_counters_by_consumer.get(root, ())
        if nested_loop_consumers:
            # Instrument the original logical loop before segmentation.
            # Split ranges then retain the original site-coordinate
            # expression instead of rebasing publication IDs per segment.
            scheduled_root_body = body_with_nested_loop_publications(
                root,
                root_body,
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
                        root_body,
                        scheduled_coordinates,
                    )
                ),
            ]
        scoped_waits.extend(barrier_waits)
        if guard_first_waits and _write_guard(tile_body) is None:
            # No guard to follow: covered waits are scoped like any root's.
            scoped_waits.extend((w, c) for w, c in guard_first_waits if c)
            guard_first_waits = [(w, c) for w, c in guard_first_waits if not c]
        state_names = frozenset(n for n in (state_arg, epoch_var) if n is not None)
        # A guarded kernel-scope root and an allocation no argument names keep
        # their waits at task entry.
        relaxed = guarded_kernel_scope_root(root)
        arguments = set(argument_allocations.values())
        for wait, covered in scoped_waits:
            if relaxed or covered - arguments:
                body.extend(wait)
            else:
                _scope_wait(
                    tile_body,
                    wait,
                    covered,
                    state_names,
                    argument_allocations,
                )
        if guard_first_waits:
            guard = _write_guard(tile_body)
            reads = (
                None if guard is None else _guard_reads(tile_body, argument_allocations)
            )
            entry = [statement for wait, _ in guard_first_waits for statement in wait]
            if reads is not None and not reads & guard_first_allocations:
                assert guard is not None
                guard.body[:0] = entry
            else:
                body[guard_first_index:guard_first_index] = entry
        tile_body[:] = _batch_waits(device_function, tile_body)
        body[:] = _batch_waits(device_function, body)
        task_body_head = body
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
        peer_counter_adds = peer_counter_publications(
            root,
            scheduled_coordinates,
            "release",
            # A worker's last task releases from the next root's first task.
            guard=f"{strategy.virtual_pid_var} != {deferred_last_task[root]}"
            if root in deferred_last_task
            else None,
        )
        if producer_counters or peer_counter_adds:
            has_task_scheduling = True
        guard = (
            _write_guard(tile_body)
            if root in kernel_scope_roots
            and producer_counters
            and not peer_counter_adds
            and root_arrival is None
            and arrives_unreleased(root)
            else None
        )
        if guard is not None:
            # A task whose guard fails writes nothing: its arrivals need no
            # release (which would wait on every outstanding write).  (A task
            # that supersedes an earlier root's access, or whose arrival runs a
            # continuation, always releases.)
            guard.orelse = _emit_arrivals(
                device_function,
                [
                    part
                    for plan, producer in producer_counters
                    for part in emit_readiness_arrivals_from_producer(
                        plan, producer, scheduled_coordinates
                    )
                ],
                "relaxed",
            )
            body = guard.body
        if producer_counters or peer_counter_adds or root_arrival is not None:
            body.extend(_release_sync(device_function))
        body.extend(
            _emit_arrivals(
                device_function,
                [
                    *(
                        part
                        for plan, producer in producer_counters
                        for part in emit_readiness_arrivals_from_producer(
                            plan, producer, scheduled_coordinates
                        )
                    ),
                    *([] if root_arrival is None else [root_arrival]),
                ],
                "release",
            )
        )
        body.extend(peer_counter_adds)
        if guard is not None:
            body = task_body_head
        if not has_task_scheduling or root in kernel_scope_roots:
            return body
        # Entry-only bookkeeping on a single-trip root may inline. Counter
        # publication and nested work retain the live-range boundary.
        is_single_trip_occurrence = (
            root not in static_pipeline_plan.continuation_roots
            and static_pipeline_plan.execution_orders[root].task_count
            <= launch_worker_count
        )
        scheduled_wrapper_noinline = (
            not is_single_trip_occurrence
            or bool(producer_counters)
            or bool(peer_counter_adds)
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

    # Live-task counts of guarded resident roots emitted at kernel scope.
    resident_live_tasks: dict[int, tuple[str, int]] = {}
    # Guard statements every worker runs at kernel scope (``_reuse_guard_values``).
    guard_statements: set[int] = set()

    deferred_last_task = {
        prev: device_function.new_var("tile_dependency_deferred_task", dce=False)
        for prev in deferred_releases.values()
    }

    def deferred_release(root: int, guard: str) -> ast.stmt:
        """Under ``guard``, this worker's last release of the previous root."""
        prev = deferred_releases[root]
        last = deferred_last_task[prev]
        coordinates = execution_coordinates(
            prev, f"{last} - {static_pipeline_plan.static_base(prev)}"
        )
        return create(
            ast.If,
            test=expr_from_string(f"({guard}) and {last} >= 0"),
            body=[
                *_release_sync(device_function),
                *peer_counter_publications(prev, coordinates, "release"),
            ],
            orelse=[],
        )

    # Where each resident root's lane 0 starts, in tasks: packed, a root starts
    # where the earlier roots' tasks end when it fits on the workers they leave
    # idle; a root that would wrap onto busy workers stays W-aligned instead (its
    # tasks then run on the workers of the earlier root's tasks they most likely
    # consume, not one rotation away).
    root_lane_origins: dict[int, int] = {}
    # The roots since the last W-aligned one: their live tasks shift a packed
    # root's start.
    packed_runs: dict[int, list[int]] = {}
    lanes_end = 0
    run: list[int] = []
    for resident in static_pipeline_plan.resident_roots:
        tasks = static_pipeline_plan.execution_orders[resident].task_count
        first_idle = lanes_end % launch_worker_count
        # A guarded extent's live tasks, and so where the roots behind it end,
        # are known only to the lanes' shift.
        if packed_placement and (
            first_idle == 0
            or first_idle + tasks <= launch_worker_count
            or resident in guarded_extents
            or any(earlier in guarded_extents for earlier in run)
        ):
            root_lane_origins[resident] = lanes_end
            packed_runs[resident] = list(run)
        else:
            root_lane_origins[resident] = static_pipeline_plan.static_base(resident)
            run = []
        run.append(resident)
        lanes_end = root_lane_origins[resident] + tasks

    def root_lane(root: int) -> str:
        # A root begins mid-wave, where the earlier roots' live tasks end
        # (without packing, only trailing roots): rotate workers onto its
        # lanes. Waves and slots stay the plan's; only a lane's worker moves.
        plan = static_pipeline_plan
        start = root_lane_origins[root]
        skipped = [
            f"({count} - tl.minimum({live}, {count}))"
            for earlier, (live, count) in resident_live_tasks.items()
            if (
                earlier in packed_runs[root]
                if root in packed_runs
                else plan.static_base(earlier) < start
            )
        ]
        if skipped:
            # Whole waves keep the dividend nonnegative.
            waves = start // launch_worker_count + 1
            return (
                f"(({worker}) + {waves * launch_worker_count - start} + "
                f"{' + '.join(skipped)}) % {launch_worker_count}"
            )
        rotation = -start % launch_worker_count
        return (
            f"(({worker}) + {rotation}) % {launch_worker_count}" if rotation else worker
        )

    def guarded_live_tasks(extent: _GuardedExtent) -> tuple[list[ast.stmt], str]:
        live = device_function.new_var("tile_dependency_live_tasks", dce=False)
        # redux.sync makes the bound provably warp-uniform, so tcgen05 issue
        # inside the task loop stays a single elected thread.
        return [
            *extent.hoisted,
            statement_from_string(
                f"{live} = tl.inline_asm_elementwise('redux.sync.min.s32 $0, $1, "
                f"0xffffffff;', '=r,r', [tl.cast({extent.live_tasks}, tl.int32)], "
                "dtype=tl.int32, is_pure=True, pack=1)"
            ),
        ], live

    def shared_captures(extent: _GuardedExtent, block_ids: list[int]) -> list[ast.stmt]:
        if not warp_specialized(block_ids):
            return []
        # Partitions share hoisted reductions and loads instead of each
        # recomputing them; cheap index math stays rematerialized.
        costly: set[str] = set()
        for statement in extent.hoisted:
            if any(
                _is_tl_call(node, _HOIST_ANCHOR_CALLS)
                or (isinstance(node, ast.Name) and node.id in costly)
                for node in ast.walk(statement)
            ):
                costly.update(_stored_names([statement]))
        used = _all_names(extent.body)
        return [
            statement_from_string(f"{name} = helion_dist_utils._shared_capture({name})")
            for name in extent.hoisted_names
            if name in used and name in costly
        ]

    def skipped_presence_word(
        store: TileAccess, mirror: TaggedMirror, coordinates: dict[int, str]
    ) -> str:
        """The row presence word of the tile a task (``coordinates``) of the
        store's root pushes into the mirror (T1: one tile per task)."""
        first: sympy.Expr = sympy.Integer(0)
        for position, dim in enumerate(store.subscript_dims):
            interval = _access_interval_expression(
                store, position=position, domain=root_domains[store.root]
            )
            if interval is None:
                raise TaggedPeerFallback(
                    f"T4: {store.tensor_name}: a skipped task's tile is unknown"
                )
            stride = _concrete_integer(
                store.tensor_strides[dim], description="tagged stride"
            )
            first = first + interval[0] * stride  # pyrefly: ignore[unsupported-operation]
        offset = relation_expression(sympy.expand(first), coordinates)
        return mirror.presence_word(f"({offset})")

    def skipped_task_dispatch(
        root: int, live: str
    ) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """The guard-skipped tasks' statements: their tagged tiles' row presence
        marks, which order nothing (a skipped task writes no rows), so they run
        ahead of the root's tasks, where the worker waits for its first; then
        their counter arrivals and scatter done words, after its tasks."""
        assert dependency_graph is not None
        scatter_root = None if peer is None else peer.scatter_roots.get(root)
        local_counters = root_counters_by_producer.get(root, ())
        # Tagged pushes of this root with row presence words to mark (T4).
        marked = [
            (allocation_id, store, mirror)
            for allocation_id, mirror in ({} if peer is None else peer.mirrors).items()
            for store in dependency_graph.tagged_stores(allocation_id)[:1]
            if store.root == root and mirror.presence is not None
        ]
        if (
            scatter_root is None
            and not local_counters
            and not marked
            and not any(
                plan.producers[0].producer_root == root for plan in peer_counter_plans
            )
        ):
            return [], []
        # Guard-skipped tasks still count once per launch for keyed peers and
        # local readiness counters, and mark their scatter done words and their
        # tagged tiles' row presence words (no rows: T4).
        segment_begin = static_pipeline_plan.static_base(root)
        segment_end = (
            segment_begin + static_pipeline_plan.execution_orders[root].task_count
        )
        lane = root_lane(root)
        skipped = device_function.new_var("tile_dependency_skipped_task", dce=False)
        first = (
            f"{segment_begin} + ({lane}) + tl.maximum({live} - ({lane}) + "
            f"{launch_worker_count - 1}, 0) // {launch_worker_count} * "
            f"{launch_worker_count}"
        )
        coordinates = execution_coordinates(root, f"({skipped}) - {segment_begin}")
        body = peer_counter_publications(root, coordinates, "relaxed")
        # A skipped task writes nothing, so its arrivals need no release.
        body.extend(
            _emit_arrivals(
                device_function,
                [
                    part
                    for plan, readiness_producer in local_counters
                    for part in emit_readiness_arrivals_from_producer(
                        plan, readiness_producer, coordinates
                    )
                ],
                "relaxed",
            )
        )
        if scatter_root is not None:
            assert peer is not None
            body.append(
                peer.every_rank_store(
                    device_function,
                    f"{scatter_root.done} + tl.cast({peer.epoch} & 1, tl.int64) * "
                    f"{world_size * scatter_root.tasks} + {peer.rank} * "
                    f"{scatter_root.tasks} + {scatter_root.task(coordinates)}",
                    f"({peer.epoch} << 32) | 0xFFFFFFFF",
                    threads,
                )
            )
        marks = [
            peer.every_rank_store(
                device_function,
                f"{peer.mirror(allocation_id)} + "
                f"{skipped_presence_word(store, mirror, coordinates)}",
                f"{peer.epoch} << 32",
                threads,
            )
            for allocation_id, store, mirror in marked
            if peer is not None
        ]

        def over_skipped(statements: list[ast.stmt]) -> list[ast.stmt]:
            if not statements:
                return []
            return [
                create(
                    ast.For,
                    target=create(ast.Name, id=skipped, ctx=ast.Store()),
                    iter=expr_from_string(
                        f"range({first}, {segment_end}, {launch_worker_count})"
                    ),
                    body=statements,
                    orelse=[],
                    type_comment=None,
                )
            ]

        return over_skipped(marks), over_skipped(body)

    def static_root_body(root: int) -> list[ast.stmt]:
        """Lower one resident root from its scalar ownership fold."""
        task_count_value = static_pipeline_plan.execution_orders[root].task_count
        active_worker_count = min(launch_worker_count, task_count_value)
        segment_begin = static_pipeline_plan.static_base(root)
        segment_end = segment_begin + task_count_value
        lane = root_lane(root)
        segment_membership = (
            f"({lane}) == 0"
            if active_worker_count == 1
            else f"(({lane}) >= 0 and ({lane}) < {active_worker_count})"
        )
        extent = guarded_extents.get(root)
        block_ids = [info.block_id for info in _case_pid_info(owner.cases[root])]
        # Claims follow a data-dependent (guarded) task count; on a static count
        # they measured costlier than the imbalance they remove.
        claimed = root in claim_indices and extent is not None
        assert not (claimed and root in deferred_last_task)
        hoisted: list[ast.stmt] = []
        skipped_marks: list[ast.stmt] = []
        skipped_dispatch: list[ast.stmt] = []
        segment_stop = f"({segment_end})"
        live: str | None = None
        if extent is not None:
            hoisted, live = guarded_live_tasks(extent)
            if active_worker_count == launch_worker_count:
                guard_statements.update(
                    id(statement)
                    for statement in hoisted
                    if isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                )
            # Captures follow the whole hoisted block so MLIR CSE still sees it.
            hoisted.extend(shared_captures(extent, block_ids))
            segment_stop = f"tl.minimum({segment_begin} + {live}, {segment_end})"
            skipped_marks, skipped_dispatch = skipped_task_dispatch(root, live)
        first_task = f"((({lane}) - 0) + ({segment_begin}))"
        has_task = f"{first_task} < {segment_stop}"
        task_arguments = (
            strategy.virtual_pid_var,
            *(() if extent is None else extent.hoisted_names),
        )
        # Kernel-scope statements around the root (outside its membership guard)
        # and the first statements of its task loop.
        kernel_scope: list[ast.stmt] = []
        trailer: list[ast.stmt] = []
        task_prologue: list[ast.stmt] = []
        if root in deferred_last_task:
            last = deferred_last_task[root]
            kernel_scope.append(statement_from_string(f"{last} = -1"))
            hoisted.append(
                statement_from_string(
                    f"{last} = tl.where({has_task}, {first_task} + ({segment_stop}"
                    f" - {first_task} - 1) // {launch_worker_count} * "
                    f"{launch_worker_count}, -1)"
                )
            )
            task_arguments = (*task_arguments, last)
        if root in deferred_releases:
            ran = device_function.new_var("tile_dependency_deferred_ran", dce=False)
            kernel_scope.append(statement_from_string(f"{ran} = 0"))
            hoisted.append(statement_from_string(f"{ran} = tl.where({has_task}, 1, 0)"))
            task_prologue.append(
                deferred_release(root, f"{strategy.virtual_pid_var} == {first_task}")
            )
            # A worker that runs no task here still owes the release.
            trailer.append(deferred_release(root, f"{ran} == 0"))
        invariant_waits: list[ast.stmt] = []
        # A root barrier ordering only argument reads the root's hoisted
        # statements make none of is placed in each task like its covered
        # readiness waits; a worker with no task (guarded extent) waits once
        # before its skipped-task arrivals and publications, which follow the
        # wait as they did at the root's entry.
        incoming_roots = root_barrier_incoming.get(root, ())
        every_worker = active_worker_count == launch_worker_count
        # None: a hoisted load reads through a name that is no tensor argument.
        hoisted_loads = _loaded_allocations(hoisted, argument_allocations)
        hoisted_reads = (
            None
            if hoisted_loads is None
            else hoisted_loads
            | {
                allocation
                for statement in hoisted
                for allocation in _named_allocations(statement, argument_allocations)
            }
        )
        barrier_waits: list[ast.stmt] = []
        task_barriers: list[tuple[list[ast.stmt], frozenset[int]]] = []
        # Producers whose barrier a worker with no task here still waits for.
        no_task_producers: list[int] = []
        for producer in incoming_roots:
            covered, publication = barrier_covers(root, producer)
            if (
                claimed
                or root in ws_roots
                or guarded_kernel_scope_root(root)
                or covered is None
                or publication
                or hoisted_reads is None
                or covered & hoisted_reads
            ):
                barrier_waits.extend(root_barrier_wait(root, producer, every_worker))
                continue
            wait = root_barrier_wait(root, producer, False)
            if not wait:
                continue
            # The barrier stays complete for the rest of the launch: a worker
            # waits (and fences) in its first task only.
            fence = _covering(
                device_function, device_function.async_load_fence(), covered
            )
            first_wait = create(
                ast.If,
                test=expr_from_string(f"{strategy.virtual_pid_var} == {first_task}"),
                body=[*wait, *fence],
                orelse=[],
            )
            task_barriers.append(([first_wait], covered))
            if extent is not None:
                no_task_producers.append(producer)
        # A worker's root-barrier arrival rides its last task's arrivals: one
        # release atomic instead of a second after the loop (each waits for
        # the previous one's acknowledgment).  Without an extent every member
        # runs a task.
        root_arrival = (
            _Arrival(
                root_barrier_counter(root),
                membership=f"({strategy.virtual_pid_var}) + {launch_worker_count}"
                f" >= {segment_stop}",
            )
            if folds_root_arrival(root)
            and root_counters_by_producer.get(root)
            and extent is None
            and root not in ws_roots
            and root not in deferred_last_task
            else None
        )
        task_body = task_prologue + scheduled_root_task_body(
            root,
            f"({strategy.virtual_pid_var}) - {segment_begin}",
            f"{case_offsets[root]} + (({strategy.virtual_pid_var}) - {segment_begin})",
            task_arguments,
            # A hoisted wait would precede the deferred release it may need.
            invariant_waits=None if root in deferred_releases else invariant_waits,
            barrier_waits=task_barriers,
            root_arrival=root_arrival,
        )
        if root in settled_extents:
            # The extent's hoisted loads read what these waits protect (every
            # worker here has a static task).
            hoisted[:0] = invariant_waits
        elif invariant_waits:
            hoisted.append(
                create(
                    ast.If,
                    test=expr_from_string(has_task),
                    body=invariant_waits,
                    orelse=[],
                )
            )
        if claimed:
            # Each worker's first task is its static one (the first wave keeps
            # the scheduler's placement); the rest are claimed. A worker with a
            # first task ends on one failing claim, so the root takes exactly
            # ``tasks`` claims, the last of which resets the counter.  A
            # failing ticket is the static task count, so the loop's test needs
            # no live count: under warp specialization only the claimer (which
            # issues the tickets) computes it, not every partition taking them.
            assert claim_counter_arg is not None
            claim = device_function.new_var("tile_dependency_claim", dce=False)
            device_function.sync_roles[claim] = SyncRole("claim")
            tasks = f"({segment_stop} - {segment_begin})"
            index = f"tl.cast({claim}, {CompileEnvironment.current().index_type()})"
            take = (
                f"{claim} = helion_dist_utils._claim({claim_counter_arg} + "
                f"{claim_indices[root] * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}, "
                f"{tasks}, {launch_worker_count}, {task_count_value})"
            )
            task_loop = create(
                ast.While,
                test=expr_from_string(f"{index} < {task_count_value}"),
                body=[
                    statement_from_string(
                        f"{strategy.virtual_pid_var} = {segment_begin} + {index}"
                    ),
                    *task_body,
                    statement_from_string(take),
                ],
                orelse=[],
            )
            hoisted.append(
                statement_from_string(
                    f"{claim} = tl.cast(tl.where({has_task}, {lane}, "
                    f"{task_count_value}), tl.uint64)"
                )
            )
        else:
            task_loop = create(
                ast.For,
                target=create(ast.Name, id=strategy.virtual_pid_var, ctx=ast.Store()),
                # The root's range config (e.g. warp_specialize) applies here.
                iter=expr_from_string(
                    TileStrategy.get_range_call_str(
                        device_function.config,
                        block_ids,
                        begin=first_task,
                        end=segment_stop,
                        step=str(launch_worker_count),
                    )
                ),
                body=task_body,
                orelse=[],
                type_comment=None,
            )
        waits = peer_waits(root)
        publications = peer_publications(root)
        # Only its skipped-task arrivals and publications rely on the barrier.
        no_task_waits = [
            create(
                ast.If,
                test=expr_from_string(f"not ({has_task})"),
                body=_covering(
                    device_function,
                    [*wait, *device_function.async_load_fence()],
                    frozenset(),
                    publication=True,
                ),
                orelse=[],
            )
            for producer in no_task_producers
            if skipped_dispatch or publications
            for wait in [root_barrier_wait(root, producer, False)]
        ]
        task_dispatch: list[ast.stmt] = [
            *hoisted,
            *skipped_marks,
            task_loop,
            *no_task_waits,
            *skipped_dispatch,
        ]
        if (
            not incoming_roots
            and root not in root_barrier_indices
            and not waits
            and not publications
            and active_worker_count == launch_worker_count
        ):
            if live is not None:
                resident_live_tasks[root] = (live, task_count_value)
            return [*kernel_scope, *task_dispatch, *trailer]

        active_body = _batch_waits(device_function, barrier_waits)
        if active_body:
            active_body.extend(
                _covering_as(
                    device_function, device_function.async_load_fence(), barrier_waits
                )
            )
        active_body.extend(waits)
        active_body.extend(task_dispatch)
        if publications:
            active_body.extend(_release_sync(device_function))
        if root_arrival is None:
            active_body.extend(
                root_barrier_publication(root, synced=bool(publications))
            )
        active_body.extend(publications)
        if every_worker:
            # Every worker is a member: the block runs at kernel scope, and its
            # live count reaches later roots' lanes.
            if live is not None:
                resident_live_tasks[root] = (live, task_count_value)
            return [*kernel_scope, *active_body, *trailer]
        return [
            *kernel_scope,
            create(
                ast.If,
                test=expr_from_string(segment_membership),
                body=active_body,
                orelse=[],
            ),
            *trailer,
        ]

    if not uses_packet_dispatch:
        assert state_arg is not None
        for root in static_pipeline_plan.resident_roots:
            body = static_root_body(root)
            # The gate precedes the first task loop, after its hoisted loads.
            first = next(
                (
                    i
                    for i, stmt in enumerate(body)
                    if isinstance(stmt, (ast.For, ast.While))
                ),
                0,
            )
            result.extend([*body[:first], *entry_gate, *body[first:]])
            entry_gate = []
        assert not entry_gate, "inband scatters need a resident root"
        _reuse_guard_values(result, guard_statements)
        writes: list[ast.stmt] = [
            statement_from_string(f"tl.store({state_arg} + {worker}, {epoch_var})")
        ]
        if done_roots:
            # The grid stays open until every peer's done roots published.
            assert peer is not None
            writes.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{worker} == 0"),
                    body=[
                        _bound_wait(
                            device_function,
                            f"helion_dist_utils._wait_at_least({peer.state} + "
                            f"{peer.done}, {peer_target(done_roots)})",
                        )
                    ],
                    orelse=[],
                )
            )
        if peer is not None:
            writes.append(
                statement_from_string(f"tl.store({peer_launch}, {peer.epoch})")
            )
        result.extend(writes)
        device_function.launch_writes.extend(writes)
    else:
        assert dispatch_ticket is not None
        dispatch: list[ast.stmt] = []
        packet_branches: list[tuple[int, bool, ast.stmt]] = []
        for root, packet_begin, packet_end in packet_ranges:
            local_task = f"({dispatch_ticket} - {packet_begin})"
            task_body = _wait_for_dependencies(
                device_function=device_function,
                dependencies=root_barrier_input_dependencies(root),
                prefix="tile_dependency_root_barrier_wait",
            )
            task_body.extend(peer_waits(root))
            folded = folds_root_arrival(root)
            task_body.extend(
                scheduled_root_task_body(
                    root,
                    local_task,
                    f"{case_offsets[root]} + {local_task}",
                    (dispatch_ticket,),
                    root_arrival=_Arrival(root_barrier_counter(root))
                    if folded
                    else None,
                )
            )
            publications = peer_publications(root)
            if publications:
                task_body.extend(_release_sync(device_function))
            if not folded:
                task_body.extend(
                    root_barrier_publication(root, synced=bool(publications))
                )
            task_body.extend(publications)
            # Kernel-scoped tensor-memory roots get no dry pass: tl.range fails to
            # compile around dot_scaled, and the while-loop form is slower.
            if waits_on_dependency(root) and root not in kernel_scope_roots:
                task_body = dry_pass_task(task_body) or task_body
            packet_branches.append(
                (
                    packet_begin,
                    root in kernel_scope_roots,
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{dispatch_ticket} >= {packet_begin} and "
                            f"{dispatch_ticket} < {packet_end}"
                        ),
                        body=task_body,
                        orelse=[],
                    ),
                )
            )

        # Kernel-scoped TMEM work cannot cross a Triton helper boundary. Keep
        # that prefix inline, while outlining one helper-safe suffix so its
        # register live ranges do not couple to the kernel-scoped roots.
        last_kernel_scope_branch = next(
            (
                index
                for index in reversed(range(len(packet_branches)))
                if packet_branches[index][1]
            ),
            None,
        )
        if last_kernel_scope_branch is not None and last_kernel_scope_branch + 1 < len(
            packet_branches
        ):
            dispatch.extend(
                branch
                for _begin, _requires_kernel_scope, branch in packet_branches[
                    : last_kernel_scope_branch + 1
                ]
            )
            suffix = packet_branches[last_kernel_scope_branch + 1 :]
            suffix_begin = suffix[0][0]
            helper_call = _outline_cross_loop_region(
                device_function,
                name_hint="tile_dependency_packet_dispatch",
                body=[branch for _begin, _requires_kernel_scope, branch in suffix],
                extra_argument_names=(
                    dispatch_ticket,
                    epoch_var,
                    *([peer.epoch] if peer is not None else []),
                ),
                noinline=True,
            )
            dispatch.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{dispatch_ticket} >= {suffix_begin}"),
                    body=[helper_call],
                    orelse=[],
                )
            )
        else:
            dispatch.extend(
                branch for _begin, _requires_kernel_scope, branch in packet_branches
            )
        if done_roots:
            # The last ticket exits only after every rank's done roots finish,
            # so the next launch cannot overwrite data a peer still reads.
            assert peer is not None
            dispatch.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{dispatch_ticket} == {packet_count - 1}"),
                    body=[
                        _bound_wait(
                            device_function,
                            f"helion_dist_utils._wait_at_least({peer.state} + "
                            f"{peer.done}, {peer_target(done_roots)})",
                        )
                    ],
                    orelse=[],
                )
            )
        result.extend(dispatch)
    if (
        tuple(_ast_fingerprint(body) for body in case_bodies)
        != opaque_case_fingerprints
    ):
        raise AssertionError(
            "tile-dependency lowering mutated an opaque source tile body"
        )
    return result
