from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.utils._sympy.functions import CeilDiv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

from .. import exc
from .._dist_utils import _resolve_process_group
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cross_loop_scheduler import CrossLoopDispatchMode
from .cross_loop_scheduler import ReadinessConsumer
from .cross_loop_scheduler import ReadinessCounterPlan
from .cross_loop_scheduler import ReadinessProducer
from .cross_loop_scheduler import build_static_pipeline_plan
from .cross_loop_scheduler import nested_wait_placement
from .device_function import TensorArg
from .device_function import TensorDescriptorArg
from .host_function import HostFunction
from .program_id import _clone_ast_value
from .program_id import _clone_stmt
from .program_id import typed_program_id
from .tile_dependency import TILE_ACCESS_META
from .tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DenseTaskOrder
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_coordinate_domains
from .tile_dependency import nested_logical_axes
from .tile_dependency import tile_dependency_site_id
from .tile_strategy import L2GroupingProgramIDs
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    from .device_function import DeviceFunction
    from .program_id import ForEachProgramID
    from .program_id import PersistentProgramIDs
    from .program_id import PIDInfo
    from .program_id import ProgramIDs
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
        *(device_function.async_load_fence() if load_fence else []),
    ]


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


def _emit_final_arrival_continuation(
    *,
    counter: str,
    target: str,
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
            test=expr_from_string(f"{previous} == {target} - 1"),
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


@dataclasses.dataclass(frozen=True)
class PeerState:
    """Symmetric uint64 state: inband mailboxes, peer counters, a done slot and
    the static pipeline's launch epoch.

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
    launch: int
    scatters: dict[int, Scatter] = dataclasses.field(default_factory=dict)
    scatter_roots: dict[int, ScatterRoot] = dataclasses.field(default_factory=dict)
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


def peer_state(device_function: DeviceFunction) -> PeerState | None:
    """Register the peer state once per kernel if any dependency crosses ranks."""
    if device_function.peer_state is not None:
        return device_function.peer_state
    graph = HostFunction.current().device_ir.tile_dependency_graph
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
        numel=str(done + 2),
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
        launch=done + 1,
        scatters=scatters,
        scatter_roots=scatter_roots,
        flag=flag,
        started=flag + 1,
        votes=flag + 1 + world_size,
    )
    return device_function.peer_state


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


def _is_tl_call(node: ast.AST, names: frozenset[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr in names
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

    Every other top-level statement must be a pure single assignment (or a rebase of
    a task-variant PID), so tasks past the bound are no-ops. Invariant loads and
    reductions are hoisted with the bound.
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

    if not all(
        isinstance(statement, (ast.Assign, ast.AugAssign))
        and (name := target(statement)) is not None
        and (isinstance(statement, ast.Assign) or name in variant_names)
        and all(
            _is_pure_call(node)
            for node in ast.walk(statement.value)
            if isinstance(node, ast.Call)
        )
        for statement in prefix
    ):
        return None
    stores: dict[str, int] = {}
    loads: dict[str, int] = {}
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name):
                counts = stores if isinstance(node.ctx, ast.Store) else loads
                counts[node.id] = counts.get(node.id, 0) + 1
    values = {
        cast("str", target(statement)): statement.value
        for statement in prefix
        if isinstance(statement, ast.Assign)
    }
    test = guard.test.id
    predicate = values.get(test)
    if (
        stores.get(test) != 1
        or loads.get(test) != 1
        or not isinstance(predicate, ast.Compare)
        or len(predicate.ops) != 1
    ):
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
    # The guarded offset must be the tile begin of the slowest axis.
    pid = slowest.pid_var
    offset_value = values.get(offset.id)
    if (
        stores.get(offset.id) != 1
        or offset_value is None
        or ast.unparse(offset_value)
        not in (f"{pid} * {slowest.block_size_var}", *([pid] * (block_size == 1)))
    ):
        return None

    def is_invariant(value: ast.expr, invariant: set[str]) -> bool:
        for node in ast.walk(value):
            if (
                isinstance(node, ast.Name)
                and node.id not in invariant
                and (node.id in stores or node.id in variant_names)
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

    invariant: set[str] = set()
    for name, value in values.items():
        if stores[name] == 1 and is_invariant(value, invariant):
            invariant.add(name)
    if bound.id not in invariant and (bound.id in stores or bound.id in variant_names):
        return None

    anchors = [
        name
        for name in invariant
        if name == bound.id
        or any(
            _is_tl_call(node, _HOIST_ANCHOR_CALLS) for node in ast.walk(values[name])
        )
    ]
    hoisted: set[str] = set()
    while anchors:
        name = anchors.pop()
        if name in hoisted:
            continue
        hoisted.add(name)
        anchors.extend(
            node.id
            for node in ast.walk(values[name])
            if isinstance(node, ast.Name) and node.id in invariant
        )
    hoisted_statements = [
        _clone_stmt(statement)
        for statement in prefix
        if isinstance(statement, ast.Assign) and target(statement) in hoisted
    ]
    remaining = [
        _clone_stmt(statement)
        for statement in prefix
        if target(statement) not in {*hoisted, test}
    ]
    tiles = bound.id if block_size == 1 else f"tl.cdiv({bound.id}, {block_size})"
    return _GuardedExtent(
        hoisted=hoisted_statements,
        body=[*remaining, *(_clone_stmt(statement) for statement in guard.body)],
        hoisted_names=tuple(
            cast("str", target(statement)) for statement in hoisted_statements
        ),
        live_tasks=tiles if inner_tasks == 1 else f"{inner_tasks} * {tiles}",
    )


def _stored_names(nodes: Iterable[ast.AST]) -> set[str]:
    return {
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


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
    dependency_graph = HostFunction.current().device_ir.tile_dependency_graph
    assert dependency_graph is not None
    if dependency_graph.crosses_ranks() and pipeline == "barrier":
        raise AssertionError("cross-rank tile dependencies cannot use 'barrier'")
    if pipeline == "barrier":
        device_function.has_barrier = True
        return owner._emit_phase_loops(strategy, device_function, total_expr)
    if pipeline not in ("static", "dynamic"):
        raise exc.InvalidConfig(f"unknown cross_loop_pipeline value {pipeline!r}")

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
        continuation_ineligible_roots=kernel_scope_roots,
        prove_nonnegative=CompileEnvironment.current().known_nonnegative,
        cross_loop_dispatch_mode=cast("CrossLoopDispatchMode", pipeline),
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
    if static_pipeline_plan.dispatch_mode != pipeline:
        raise AssertionError("pipeline plan disagrees with configured dispatch mode")
    uses_packet_dispatch = pipeline == "dynamic"
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
    readiness_counter_state_offset = reserve_state(readiness_counter_count)
    root_barrier_state_offset = reserve_state(root_barrier_count)
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

    def peer_arrivals(root: int) -> int:
        # Dynamic tasks publish one by one, static workers once per root.
        tasks = int(static_pipeline_plan.execution_orders[root].task_count)
        return tasks if pipeline == "dynamic" else min(tasks, launch_worker_count)

    def peer_target(roots: Iterable[int]) -> str:
        assert peer is not None
        arrivals = sum(peer_arrivals(root) for root in roots)
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
                statement_from_string(
                    f"helion_dist_utils._wait_at_least({peer.state} + "
                    f"{peer.slots[producer]}, {peer_target((producer,))})"
                )
                for producer in producers
            ),
            _publication_sync(device_function),
            *device_function.async_load_fence(),
        ]

    lanes = 1 << (world_size - 1).bit_length()

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
        root: int, coordinates: dict[int, str], sem: str
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
        return result

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
        return statements, tokens

    dispatch_ticket: str | None = None
    peer_launch = ""
    entry_gate: list[ast.stmt] = []
    if not uses_packet_dispatch:
        assert state_arg is not None
        result: list[ast.stmt] = [
            statement_from_string(f"{epoch_var} = tl.load({state_arg} + {worker}) + 1")
        ]
        if peer is not None:
            # uint64 so peer targets never wrap. Worker 0 advances it after the
            # done barrier, or else every worker counts its own launches.
            peer_launch = f"{peer.state} + {peer.launch}"
            if not done_roots:
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
                            statement_from_string(
                                "helion_dist_utils._store_on_every_rank("
                                f"{peer.ptrs}, {peer.started} + {peer.rank}, "
                                f"{peer.epoch}, {world_size}, {lanes})"
                            )
                        ],
                        orelse=[],
                    ),
                    statement_from_string(
                        f"helion_dist_utils._hold_after_fallback({peer.state}, "
                        f"{peer.started}, {world_size}, {peer.epoch}, {held})"
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
        result = [
            statement_from_string(
                f"{raw_dispatch_ticket} = tl.atomic_add("
                f"{dispatch_ticket_arg}, 1, sem='relaxed', scope='gpu')"
            ),
            statement_from_string(
                f"{dispatch_ticket} = tl.cast("
                f"{raw_dispatch_ticket} % tl.cast("
                f"{packet_count}, tl.uint64), tl.int32)"
            ),
            statement_from_string(
                f"{epoch_var} = tl.cast("
                f"{raw_dispatch_ticket} // tl.cast("
                f"{packet_count}, tl.uint64) + 1, tl.uint32)"
            ),
        ]
        if peer is not None:
            result.append(
                statement_from_string(
                    f"{peer.epoch} = {raw_dispatch_ticket} // "
                    f"tl.cast({packet_count}, tl.uint64) + 1"
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

    def root_barrier_publication(root: int, *, synced: bool = False) -> list[ast.stmt]:
        if root not in root_barrier_indices:
            return []
        barrier_counter = root_barrier_counter(root)
        arrivals = static_pipeline_plan.root_barrier_arrival_count(root)
        result = [] if synced else _release_sync(device_function)
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
    guarded_extents: dict[int, _GuardedExtent] = {}
    for root in () if uses_packet_dispatch else static_pipeline_plan.resident_roots:
        axis_order, axis_counts, block_sizes = case_geometries[root]
        inner_counts = [axis_counts[block_id] for block_id in axis_order[:-1]]
        if (
            root in scheduled_task_roots
            or root in root_counters_by_producer
            or root in nested_producer_roots
            or root in nested_loop_counters_by_consumer
            or isinstance(owner.cases[root], L2GroupingProgramIDs)
            or not all(isinstance(count, int) for count in inner_counts)
        ):
            continue
        extent = _guarded_extent(
            case_bodies[root],
            slowest=_case_pid_info(owner.cases[root])[-1],
            block_size=block_sizes[axis_order[-1]],
            inner_tasks=math.prod(cast("list[int]", inner_counts)),
            variant_names=frozenset({owner.shared_pid_var, strategy.virtual_pid_var}),
            read_only_tensors=access_names[False] - access_names[True],
            tensor_names=frozenset(
                argument.name
                for argument in device_function.arguments
                if isinstance(argument, TensorArg)
            ),
        )
        if extent is not None:
            guarded_extents[root] = extent
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
                waits = _wait_for_counter(
                    device_function=device_function,
                    counter=readiness_counter(plan, readiness_key),
                    target=readiness_target(plan, readiness_key),
                    prefix="tile_dependency_nested_loop_wait",
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
                    _wait_for_counter(
                        device_function=device_function,
                        counter=readiness_counter(plan, readiness_key),
                        target=readiness_target(plan, readiness_key),
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
                ),
            ]

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
            statement_from_string(f"{consumer_task} = {consumer_task_expression}")
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

        last_arrival_body = [*device_function.async_load_fence(), consumer_call]
        if consumer_publications:
            last_arrival_body.extend(_release_sync(device_function))
            last_arrival_body.extend(consumer_publications)
        last_arrival_body.extend(
            root_barrier_publication(
                continuation_root, synced=bool(consumer_publications)
            )
        )
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
                target=readiness_target(plan, readiness_key),
                previous=previous,
                continuation_body=last_arrival_body,
            ),
        ]

    def emit_readiness_arrivals_from_producer(
        plan: ReadinessCounterPlan,
        readiness_producer: ReadinessProducer,
        producer_coordinates: dict[int, str],
    ) -> list[ast.stmt]:
        publication = readiness_producer.incidence.keys_by_item
        if publication is None:
            raise AssertionError("readiness publication relation is unavailable")
        readiness_key, membership = relation_flat_target(
            publication,
            producer_coordinates,
            trusted_single_valued=True,
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

            publications: list[ast.stmt] = []
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

    def scheduled_root_task_body(
        root: int,
        root_local_pid_task: str,
        logical_pid: str,
        extra_argument_names: tuple[str, ...],
        *,
        force_noinline: bool = False,
    ) -> list[ast.stmt]:
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
                target=readiness_target(
                    incoming_readiness_counter,
                    readiness_key,
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
        peer_counter_spins, peer_tokens = peer_counter_waits(
            root, scheduled_coordinates
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
            root, scheduled_coordinates, "release"
        )
        if producer_counters or peer_counter_adds:
            has_task_scheduling = True
            body.extend(_release_sync(device_function))
        for producer_counter_plan, readiness_producer in producer_counters:
            body.extend(
                emit_readiness_arrivals_from_producer(
                    producer_counter_plan,
                    readiness_producer,
                    scheduled_coordinates,
                )
            )
        body.extend(peer_counter_adds)
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

    def root_lane(root: int) -> str:
        # A trailing root may begin mid-wave: rotate workers onto its lanes.
        rotation = -static_pipeline_plan.static_base(root) % launch_worker_count
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

    def skipped_task_dispatch(root: int, live: str) -> list[ast.stmt]:
        scatter_root = None if peer is None else peer.scatter_roots.get(root)
        if scatter_root is None and not any(
            plan.producers[0].producer_root == root for plan in peer_counter_plans
        ):
            return []
        # Guard-skipped tasks still count once per launch for keyed peers and
        # mark their scatter done words.
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
        if scatter_root is not None:
            assert peer is not None
            body.append(
                statement_from_string(
                    f"helion_dist_utils._store_on_every_rank({peer.ptrs}, "
                    f"{scatter_root.done} + tl.cast({peer.epoch} & 1, tl.int64) * "
                    f"{world_size * scatter_root.tasks} + {peer.rank} * "
                    f"{scatter_root.tasks} + {scatter_root.task(coordinates)}, "
                    f"({peer.epoch} << 32) | 0xFFFFFFFF, {world_size}, {lanes})"
                )
            )
        return [
            create(
                ast.For,
                target=create(ast.Name, id=skipped, ctx=ast.Store()),
                iter=expr_from_string(
                    f"range({first}, {segment_end}, {launch_worker_count})"
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
        ]

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
        hoisted: list[ast.stmt] = []
        skipped_dispatch: list[ast.stmt] = []
        segment_stop = f"({segment_end})"
        if extent is not None:
            hoisted, live = guarded_live_tasks(extent)
            segment_stop = f"tl.minimum({segment_begin} + {live}, {segment_end})"
            skipped_dispatch = skipped_task_dispatch(root, live)
        task_dispatch: list[ast.stmt] = [
            *hoisted,
            create(
                ast.For,
                target=create(
                    ast.Name,
                    id=strategy.virtual_pid_var,
                    ctx=ast.Store(),
                ),
                # The root's range config (e.g. warp_specialize) applies here.
                iter=expr_from_string(
                    TileStrategy.get_range_call_str(
                        device_function.config,
                        [info.block_id for info in _case_pid_info(owner.cases[root])],
                        begin=f"(({lane}) - 0) + ({segment_begin})",
                        end=segment_stop,
                        step=str(launch_worker_count),
                    )
                ),
                body=scheduled_root_task_body(
                    root,
                    f"({strategy.virtual_pid_var}) - {segment_begin}",
                    f"{case_offsets[root]} + "
                    f"(({strategy.virtual_pid_var}) - {segment_begin})",
                    (
                        strategy.virtual_pid_var,
                        *(() if extent is None else extent.hoisted_names),
                    ),
                ),
                orelse=[],
                type_comment=None,
            ),
            *skipped_dispatch,
        ]
        incoming_roots = root_barrier_incoming.get(root, ())
        waits = peer_waits(root)
        publications = peer_publications(root)
        if (
            not incoming_roots
            and root not in root_barrier_indices
            and not waits
            and not publications
            and active_worker_count == launch_worker_count
        ):
            return task_dispatch

        active_body = _wait_for_dependencies(
            device_function=device_function,
            dependencies=root_barrier_input_dependencies(root),
            prefix="tile_dependency_root_barrier_wait",
        )
        active_body.extend(waits)
        active_body.extend(task_dispatch)
        if publications:
            active_body.extend(_release_sync(device_function))
        active_body.extend(root_barrier_publication(root, synced=bool(publications)))
        active_body.extend(publications)
        return [
            create(
                ast.If,
                test=expr_from_string(segment_membership),
                body=active_body,
                orelse=[],
            )
        ]

    if not uses_packet_dispatch:
        assert state_arg is not None
        for root in static_pipeline_plan.resident_roots:
            body = static_root_body(root)
            # The gate precedes the first task loop, after its hoisted loads.
            first = next(
                (i for i, stmt in enumerate(body) if isinstance(stmt, ast.For)), 0
            )
            result.extend([*body[:first], *entry_gate, *body[first:]])
            entry_gate = []
        assert not entry_gate, "inband scatters need a resident root"
        result.append(
            statement_from_string(f"tl.store({state_arg} + {worker}, {epoch_var})")
        )
        if done_roots:
            # Every worker that reads the epoch publishes to the done slot first.
            assert peer is not None
            result.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{worker} == 0"),
                    body=[
                        statement_from_string(
                            f"helion_dist_utils._wait_at_least({peer.state} + "
                            f"{peer.done}, {peer_target(done_roots)})"
                        ),
                        statement_from_string(f"tl.store({peer_launch}, {peer.epoch})"),
                    ],
                    orelse=[],
                )
            )
        elif peer is not None:
            result.append(
                statement_from_string(f"tl.store({peer_launch}, {peer.epoch})")
            )
    else:
        assert dispatch_ticket is not None
        packet_branches: list[tuple[int, bool, ast.stmt]] = []
        for root, packet_begin, packet_end in packet_ranges:
            local_task = f"({dispatch_ticket} - {packet_begin})"
            task_body = _wait_for_dependencies(
                device_function=device_function,
                dependencies=root_barrier_input_dependencies(root),
                prefix="tile_dependency_root_barrier_wait",
            )
            task_body.extend(peer_waits(root))
            task_body.extend(
                scheduled_root_task_body(
                    root,
                    local_task,
                    f"{case_offsets[root]} + {local_task}",
                    (dispatch_ticket,),
                )
            )
            publications = peer_publications(root)
            if publications:
                task_body.extend(_release_sync(device_function))
            task_body.extend(root_barrier_publication(root, synced=bool(publications)))
            task_body.extend(publications)
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
            result.extend(
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
            result.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{dispatch_ticket} >= {suffix_begin}"),
                    body=[helper_call],
                    orelse=[],
                )
            )
        else:
            result.extend(
                branch for _begin, _requires_kernel_scope, branch in packet_branches
            )
        if done_roots:
            # The last ticket exits only after every rank's done roots finish,
            # so the next launch cannot overwrite data a peer still reads.
            assert peer is not None
            result.append(
                create(
                    ast.If,
                    test=expr_from_string(f"{dispatch_ticket} == {packet_count - 1}"),
                    body=[
                        statement_from_string(
                            f"helion_dist_utils._wait_at_least({peer.state} + "
                            f"{peer.done}, {peer_target(done_roots)})"
                        )
                    ],
                    orelse=[],
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
