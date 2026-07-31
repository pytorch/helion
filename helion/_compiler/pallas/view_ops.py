"""Pallas-backend codegen for ops defined in ``helion.language.view_ops``.

This module also implements *resident subviews*.  When a tile load's consumers
all narrow one dimension to a contiguous run -- ``block = x[outer, :, :]``
followed by ``block[local.begin, :, :]`` inside a nested loop -- the load is
emitted as a Pallas Ref rather than a materialized array, and each consumer
reads a small run out of that Ref.

The load node stays where it is, in the outer loop body, so the block is still
staged and DMA double-buffered once per outer tile; only the read of an
individual run moves into the inner loop.

This is a lowering, not an optimization.  Mosaic has no ``dynamic_slice`` and
no gather, so a run selected out of a value already in vector registers has no
lowering at all; the Ref is the only way to express one.  A narrowing subscript
that cannot be planned is therefore rejected with the reason, rather than
lowered to something that only works under the Pallas interpreter.  Plain
``None``/``:`` subscripts do not narrow anything and keep their ordinary value
lowering.

Planning runs once per config, because eligibility depends on the configured
block sizes.  It normalizes each narrowing subscript into exactly one
``_Selector`` and attaches it to the node; codegen consumes that and never
re-derives it.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language import _tracing_ops
from ...language.memory_ops import load
from ...language.tile_ops import tile_index
from ...language.view_ops import join
from ...language.view_ops import split
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr

if TYPE_CHECKING:
    import ast

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

# Set on a ``load`` node whose block must be emitted as a Pallas Ref.
_LOAD_AS_REF = "pallas_load_as_ref"
# Set on a narrowing ``subscript`` node: the ``_Subview`` that lowers it.
_SUBVIEW = "pallas_subview"
# Set on a narrowing ``subscript`` node that could not be planned: why.
_REJECTION = "pallas_subview_rejection"


@dataclasses.dataclass(frozen=True)
class _Block:
    """The tile of one dimension that a resident load brings into VMEM."""

    dim: int
    """Dimension of the tensor the load tiles."""

    block_id: int
    """Tile loop that produced the block."""

    extent: int
    """Size of the block along ``dim`` under this config."""

    whole_dim: bool
    """The tile loop sweeps the tensor dimension end to end, so every row of
    the block holds real data and narrowing reads need no range masking."""


class _SelectorKind(enum.Enum):
    """How a subview picks the run it reads out of a block."""

    CONSTANT = enum.auto()
    """A literal index or slice."""

    TILE = enum.auto()
    """An inner tile loop's offset, one block wide."""

    DYNAMIC = enum.auto()
    """A data-dependent start with a constant width, as written by
    ``block[live - 3 : live]``."""


@dataclasses.dataclass(frozen=True)
class _Selector:
    """One contiguous run narrowed out of a single dimension of a block.

    Produced once by ``_normalize`` during planning.  Everything codegen needs
    is either here or reachable from the block id, so codegen never re-inspects
    the index.
    """

    kind: _SelectorKind

    dim: int
    """Dimension of the block being narrowed."""

    width: int
    """Length of the run.  For ``TILE`` this is the inner loop's block size."""

    block_extent: int
    """Size of the block along ``dim``, so a data-dependent start can be
    clamped to a run that stays inside it."""

    begin: int = 0
    """Literal start of the run.  Only used by ``CONSTANT``."""

    block_id: int = -1
    """Inner tile loop driving the run.  Only used by ``TILE``."""

    dynamic_begin: object = None
    """Start of a ``DYNAMIC`` run, as the traced ``SymInt``.  Codegen renders
    it; it does not go back to the node to find it."""

    squeeze: bool = False
    """The narrowed dimension does not appear in the result."""

    masked: bool = False
    """The run can extend past the rows the tile actually covers, so the read
    has to be multiplied by the driving loop's mask."""


@dataclasses.dataclass(frozen=True)
class _Subview:
    """A ``subscript`` that reads a contiguous run out of a resident load."""

    source: torch.fx.Node
    selector: _Selector


def _node_value(value: object) -> object:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _input_rank(node: torch.fx.Node) -> int:
    """Rank of the value a ``subscript`` narrows, or -1 if it is not a tensor."""
    value = _node_value(node.args[0])
    return value.ndim if isinstance(value, torch.Tensor) else -1


def _where(node: torch.fx.Node) -> str:
    """Describe where a node came from, for an error the user can act on."""
    location = node.meta.get("location")
    filename = getattr(location, "filename", None)
    lineno = getattr(location, "lineno", None)
    if filename is None or lineno is None:
        return f"<{node.name}>"
    return f"{filename}:{lineno}"


def subview_of(node: torch.fx.Node) -> _Subview | None:
    """Return the plan attached to a narrowing ``subscript``, if it has one."""
    plan = node.meta.get(_SUBVIEW)
    return plan if isinstance(plan, _Subview) else None


def load_is_ref(node: torch.fx.Node) -> bool:
    """True when a ``load``'s block must be emitted as a Pallas Ref."""
    return node.meta.get(_LOAD_AS_REF) is True


def narrowing_positions(node: torch.fx.Node) -> list[int]:
    """Positions of a ``subscript``'s indices that select a subset of a dim.

    ``None`` adds a dimension and ``:`` keeps one whole; everything else picks
    out part of a dimension and needs a resident block to read from.
    """
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)):
        return []
    return [
        position
        for position, index in enumerate(indices)
        if index is not None and index != slice(None)
    ]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _control_flow_parents(
    graphs: list[GraphInfo],
) -> dict[int, tuple[torch.fx.Node, int]]:
    """Map each subgraph id to the node that runs it and its capture-arg index.

    ``_for_loop(graph_id, begin, end, args)`` and
    ``_for_loop_step(graph_id, begin, end, args, step)`` pass captures in
    ``args[3]``; ``_if(test, if_graph_id, else_graph_id, if_args, else_args)``
    passes them in ``args[3]`` and ``args[4]``.
    """
    parents: dict[int, tuple[torch.fx.Node, int]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if _tracing_ops.is_for_loop_target(node.target) and isinstance(
                node.args[0], int
            ):
                parents[node.args[0]] = (node, 3)
            elif node.target is _tracing_ops._if and len(node.args) >= 5:
                if isinstance(node.args[1], int):
                    parents[node.args[1]] = (node, 3)
                if isinstance(node.args[2], int):
                    parents[node.args[2]] = (node, 4)
    return parents


def _capture_map(
    graphs: list[GraphInfo], parents: dict[int, tuple[torch.fx.Node, int]]
) -> dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]]:
    """Map an outer node to the ``(placeholder, parent)`` pairs that bind it.

    A subgraph's placeholders were traced from exactly the list its parent node
    carries -- ``DeviceIR.add_graph(node_args=inputs.get_node_args(tracer))`` and
    the parent's capture argument are both built from the same
    ``inputs.get_tensor_args()`` -- so position ``i`` of one always corresponds
    to position ``i`` of the other.  This is the correspondence
    ``NodeArgsGraphInfo.placeholder_to_outer_arg`` relies on, read off the
    per-config graph copies rather than the originals.
    """
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]] = {}
    for info in graphs:
        entry = parents.get(info.graph_id)
        if entry is None:
            continue
        parent, arg_index = entry
        outer_args = parent.args[arg_index] if arg_index < len(parent.args) else None
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        if not isinstance(outer_args, (list, tuple)) or len(outer_args) != len(
            placeholders
        ):
            continue
        for outer, placeholder in zip(outer_args, placeholders, strict=True):
            if isinstance(outer, torch.fx.Node):
                captures.setdefault(outer, []).append((placeholder, parent))
    return captures


def _effective_users(
    node: torch.fx.Node,
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
) -> list[torch.fx.Node]:
    """Users of ``node`` after seeing through renames and loop captures.

    ``_new_var`` and capture placeholders only rebind the value to another
    name, which a Pallas Ref survives unchanged, so they are transparent here.
    """
    results: list[torch.fx.Node] = []
    seen: set[torch.fx.Node] = {node}
    stack: list[torch.fx.Node] = [node]
    while stack:
        current = stack.pop()
        edges = captures.get(current, ())
        consumers = {parent for _placeholder, parent in edges}
        for user in current.users:
            if user in consumers or user.target is torch.ops.aten.sym_size.int:
                continue
            if user.target is _tracing_ops._new_var:
                if user not in seen:
                    seen.add(user)
                    stack.append(user)
            else:
                results.append(user)
        for placeholder, _parent in edges:
            if placeholder not in seen:
                seen.add(placeholder)
                stack.append(placeholder)
    return results


def _resolve_index(
    node: object, placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node | None:
    """Follow renames and loop captures back to the node defining an index.

    An index written in one scope and used in another arrives as a chain of
    ``_new_var`` renames and capture placeholders; the node underneath is what
    tells a tile run apart from an arbitrary gather.
    """
    seen: set[torch.fx.Node] = set()
    while isinstance(node, torch.fx.Node) and node not in seen:
        seen.add(node)
        if node.target is _tracing_ops._new_var and node.args:
            node = node.args[0]
        elif node.op == "placeholder" and node in placeholder_to_outer:
            node = placeholder_to_outer[node]
        else:
            return node
    return None


def _mutated_tensor_ids(graphs: list[GraphInfo]) -> set[int]:
    """Ids of fake tensors written by a store or atomic anywhere on device.

    A resident load reads its Ref at the consumer's program point instead of at
    the load, so a write to the same tensor in between would be seen by reads
    that must observe the pre-write data.  Ruling out every written tensor is
    coarse, but it keeps the rule simple to state and cheap to check.
    """
    from ...language.atomic_ops import ATOMIC_OPS
    from ...language.memory_ops import store

    targets = ATOMIC_OPS | {store}
    mutated: set[int] = set()
    for info in graphs:
        for node in info.graph.nodes:
            if node.op == "call_function" and node.target in targets and node.args:
                value = _node_value(node.args[0])
                if isinstance(value, torch.Tensor):
                    mutated.add(id(value))
    return mutated


def _loop_bounds(
    graph_info: GraphInfo, parent: torch.fx.Node | None, block_id: int
) -> tuple[object, object] | None:
    """Return the ``(begin, end)`` a for-loop graph iterates ``block_id`` over."""
    from ..device_ir import ForLoopGraphInfo

    if (
        not isinstance(graph_info, ForLoopGraphInfo)
        or parent is None
        or block_id not in graph_info.block_ids
    ):
        return None
    position = graph_info.block_ids.index(block_id)
    begins, ends = parent.args[1:3]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    return _node_value(begins[position]), _node_value(ends[position])


def _enclosing_loop_bounds(
    graph_info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    block_id: int,
) -> tuple[object, object] | None:
    """Find the bounds of the tile loop enclosing ``graph_info``.

    The loop need not be the immediately enclosing graph: a subscript inside an
    ``if`` inside a tile loop is still driven by that loop, so the parent chain
    is walked until the loop that owns ``block_id`` turns up.
    """
    seen: set[int] = set()
    info: GraphInfo | None = graph_info
    while info is not None and id(info) not in seen:
        seen.add(id(info))
        entry = parents.get(info.graph_id)
        parent = entry[0] if entry is not None else None
        bounds = _loop_bounds(info, parent, block_id)
        if bounds is not None:
            return bounds
        info = graph_infos.get(parent.graph) if parent is not None else None
    return None


def _resident_block(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parent: torch.fx.Node | None,
    config: Config,
    mutated: set[int],
) -> _Block | str:
    """Describe the block a ``load`` brings in, or say why it cannot be one.

    A resident load tiles exactly one dimension and passes every other one
    through, so its result keeps the tensor's rank and a ``pl.ds`` run of the
    Ref addresses the same elements the materialized array would have held.
    """
    from .backend import SliceAddressing
    from .backend import _slice_addressing
    from .plan_tiling import IndirectGatherPattern

    tensor = _node_value(node.args[0])
    value = node.meta.get("val")
    indices = node.args[1]
    if (
        not isinstance(tensor, torch.Tensor)
        or not isinstance(value, torch.Tensor)
        or not isinstance(indices, (list, tuple))
        or node.args[2] is not None
    ):
        return "it is not a plain tile load"
    if tensor.ndim != value.ndim:
        return "it does not keep the tensor's rank"
    if id(tensor) in mutated:
        return "the tensor it reads is also written on device"
    if any(
        isinstance(pattern, IndirectGatherPattern)
        for pattern in node.meta.get("indexing_patterns") or ()
    ):
        return "it is an indirect gather"

    tiled = [dim for dim, index in enumerate(indices) if index != slice(None)]
    if len(tiled) != 1:
        return f"it tiles {len(tiled)} dimensions, not exactly one"
    dim = tiled[0]

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(_node_value(indices[dim]))
    if block_id is None:
        return f"dimension {dim} is not indexed by a tile"
    extent = env.block_sizes[block_id].from_config(config)
    if not isinstance(extent, int):
        return f"dimension {dim} has no concrete block size"
    lane_extent = env.size_hint(value.shape[-1])
    if _slice_addressing(value, dim, lane_extent) is not SliceAddressing.DIRECT:
        return f"a run on dimension {dim} would need sublane alignment"

    bounds = _loop_bounds(graph_info, parent, block_id)
    whole_dim = False
    if bounds is not None:
        begin, end = bounds
        whole_dim = (
            isinstance(begin, (int, torch.SymInt))
            and isinstance(end, (int, torch.SymInt))
            and env.known_equal(begin, 0)
            and env.known_equal(end, tensor.shape[dim])
        )
    return _Block(dim, block_id, extent, whole_dim)


def _bounded_by_block(value: torch.SymInt, block_id: int) -> bool:
    """True when ``value`` is the live extent of the tile owning ``block_id``.

    A loop running to that extent only visits rows the outer tile really
    covers, which is what lets a narrowing read skip the range mask the outer
    load would otherwise have applied.
    """
    from ..type_info import _detect_outer_block_bound

    env = CompileEnvironment.current()
    if env.get_block_id(value) is not None:
        return False
    return _detect_outer_block_bound(value, env) == block_id


def _tile_run(index: torch.fx.Node) -> int | None:
    """Return the block id of a whole, unshifted tile run, else None.

    ``tile.index`` may reach the subscript as an arithmetic node -- ``tile.index
    + 0`` traces to an ``add`` -- so the run is recognized from the
    ``tile_with_offset`` provenance the device IR already records rather than
    from the node's target.  A *shifted* run is refused: the loop's mask marks
    which of ``offset + arange(width)`` are live, which is not the right mask
    for rows read at ``offset + shift + arange(width)``.
    """
    from ..indexing_strategy import subscript_tile_info

    env = CompileEnvironment.current()
    info = subscript_tile_info(env, index)
    if info is not None:
        return info.block_id if env.known_equal(info.offset, 0) else None
    # ``subscript_tile_info`` resolves a block id from an index's value, which
    # a bare ``tile_index`` node does not carry -- its value is the run itself.
    if index.target is tile_index and index.args:
        return env.resolve_block_id(_node_value(index.args[0]))
    return None


def _normalize_tile(
    index: torch.fx.Node,
    graph_info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    config: Config,
    block: _Block,
) -> _Selector | str:
    """Normalize a ``tile.begin`` position or a ``tile.index`` run."""
    from ..host_function import HostFunction
    from ..variable_origin import TileBeginOrigin

    env = CompileEnvironment.current()
    index_value = index.meta.get("val")
    squeeze = isinstance(index_value, torch.SymInt)
    if squeeze:
        expr = _symint_expr(index_value)
        origin = (
            HostFunction.current().expr_to_origin.get(expr)
            if expr is not None
            else None
        )
        if origin is None or not isinstance(origin.origin, TileBeginOrigin):
            return "the position is not a tile's begin"
        inner_block_id = origin.origin.block_id
    else:
        run = _tile_run(index)
        if run is None:
            return (
                "the index is not a whole, unshifted tile run; a shifted run "
                "or a computed row vector would be a gather"
            )
        inner_block_id = run

    bounds = _enclosing_loop_bounds(graph_info, parents, graph_infos, inner_block_id)
    if bounds is None:
        return "no enclosing tile loop drives the index"
    begin, end = bounds
    if not isinstance(begin, (int, torch.SymInt)) or not env.known_equal(begin, 0):
        return "the driving tile loop does not start at zero"

    width = env.block_sizes[inner_block_id].from_config(config)
    if not isinstance(width, int) or width < 1:
        return "the driving tile loop has no concrete block size"
    if squeeze and width != 1:
        return f"a single position needs block size 1, but the loop uses {width}"
    if width > block.extent:
        return (
            f"the run is {width} wide but the block holds only {block.extent} "
            "rows for this config"
        )
    if block.extent % width != 0:
        return (
            f"a {width}-wide run does not divide the {block.extent}-row block "
            "for this config, so the last one would overrun it"
        )

    if isinstance(end, int):
        # A constant trip count only stays within live rows when the whole
        # block is live and the loop covers exactly it.
        if not block.whole_dim or end != block.extent:
            return "the loop's constant trip count does not cover exactly the block"
    elif isinstance(end, torch.SymInt):
        if not block.whole_dim and not _bounded_by_block(end, block.block_id):
            return (
                "the loop is not bounded by this tile's live extent, so the "
                "run could read rows the load masked away"
            )
    else:
        return "the driving tile loop has no usable end"

    return _Selector(
        _SelectorKind.TILE,
        block.dim,
        width,
        block.extent,
        block_id=inner_block_id,
        squeeze=squeeze,
        masked=not squeeze,
    )


def _normalize_iota(index: torch.fx.Node, block: _Block) -> _Selector | str:
    """Normalize a constant-width run at a data-dependent start.

    ``block[live - 3 : live]`` traces to an ``iota`` of a constant length, so
    the width is static even though the start is not.  The start is clamped
    into the block at codegen, which is what ``jax.lax.dynamic_slice`` does
    with an out-of-range start, so a guarded and an unguarded read agree.
    """
    width = _node_value(index.args[0]) if index.args else None
    start = _node_value(index.kwargs.get("start"))
    if index.kwargs.get("step") != 1:
        return "the run has a step other than one"
    if not isinstance(width, int) or width < 1:
        return "the run has no constant width"
    if width > block.extent:
        return (
            f"the run is {width} wide but the block holds only {block.extent} "
            "rows for this config"
        )
    if not isinstance(start, (int, torch.SymInt)):
        return "the run has no usable start"
    if not block.whole_dim:
        return (
            "the block is only partly live, so a data-dependent start could "
            "read rows the load masked away"
        )
    return _Selector(
        _SelectorKind.DYNAMIC, block.dim, width, block.extent, dynamic_begin=start
    )


def _normalize(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parents: dict[int, tuple[torch.fx.Node, int]],
    graph_infos: dict[torch.fx.Graph, GraphInfo],
    config: Config,
    block: _Block,
    placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node],
) -> _Selector | str:
    """Normalize one narrowing ``subscript`` of ``block``, or say why not.

    This is the only place an index is recognized.  Codegen consumes the
    ``_Selector`` this returns and never inspects the index again.
    """
    indices = node.args[1]
    assert isinstance(indices, (list, tuple))
    if any(index is None for index in indices):
        return "it also adds a dimension, which a resident run cannot do"
    positions = narrowing_positions(node)
    if len(positions) != 1:
        return f"it narrows {len(positions)} dimensions, not exactly one"
    # With no ``None`` entries the subscript positions are the block's
    # dimensions, so the narrowed position has to be the tiled one.
    position = positions[0]
    if position != block.dim:
        return (
            f"it narrows dimension {position}, but the block is resident along "
            f"dimension {block.dim}"
        )
    index = indices[position]

    if isinstance(index, int):
        if index >= block.extent:
            return f"index {index} is past the {block.extent}-row block for this config"
        if not block.whole_dim:
            return "the block is only partly live, so a fixed row may be masked away"
        return _Selector(
            _SelectorKind.CONSTANT,
            block.dim,
            1,
            block.extent,
            begin=index,
            squeeze=True,
        )

    if isinstance(index, slice):
        if index.step not in (None, 1):
            return "the run has a step other than one"
        if not isinstance(index.start, int) or not isinstance(index.stop, int):
            return "the run does not have constant bounds"
        if index.stop > block.extent:
            return (
                f"the run reaches row {index.stop} but the block holds only "
                f"{block.extent} rows for this config"
            )
        if not block.whole_dim:
            return "the block is only partly live, so a fixed run may be masked away"
        return _Selector(
            _SelectorKind.CONSTANT,
            block.dim,
            index.stop - index.start,
            block.extent,
            begin=index.start,
        )

    if not isinstance(index, torch.fx.Node):
        return f"the index type {type(index).__name__} is not supported"
    source = _resolve_index(index, placeholder_to_outer) or index
    if source.target is torch.ops.prims.iota.default:
        return _normalize_iota(source, block)
    return _normalize_tile(source, graph_info, parents, graph_infos, config, block)


def plan_resident_subviews(graphs: list[GraphInfo], config: Config) -> None:
    """Plan resident loads and their subviews, then reject what did not plan.

    Runs per config: eligibility depends on the configured block sizes, so a
    subscript can plan under one config and not another.
    """
    for info in graphs:
        for node in info.graph.nodes:
            node.meta.pop(_LOAD_AS_REF, None)
            node.meta.pop(_SUBVIEW, None)
            node.meta.pop(_REJECTION, None)

    parents = _control_flow_parents(graphs)
    captures = _capture_map(graphs, parents)
    placeholder_to_outer = {
        placeholder: outer
        for outer, edges in captures.items()
        for placeholder, _parent in edges
    }
    graph_infos = {info.graph: info for info in graphs}
    mutated = _mutated_tensor_ids(graphs)

    def parent_of(info: GraphInfo) -> torch.fx.Node | None:
        entry = parents.get(info.graph_id)
        return entry[0] if entry is not None else None

    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or node.target is not load:
                continue
            users = _effective_users(node, captures)
            narrowing = [
                user
                for user in users
                if user.op == "call_function"
                and user.target is subscript
                and narrowing_positions(user)
            ]
            if not narrowing:
                continue

            block = _resident_block(node, info, parent_of(info), config, mutated)
            if isinstance(block, str):
                for user in narrowing:
                    user.meta[_REJECTION] = (
                        f"the block it reads cannot stay resident because {block}"
                    )
                continue

            # A consumer that takes the whole block forces it to materialize,
            # which leaves no Ref for the narrowing consumers to read.
            blockers = [
                user
                for user in users
                if user not in narrowing or _input_rank(user) != node.meta["val"].ndim
            ]
            if blockers:
                where = ", ".join(sorted({_where(user) for user in blockers[:3]}))
                for user in narrowing:
                    user.meta[_REJECTION] = (
                        "the block it reads is also consumed whole at "
                        f"{where}, so it cannot stay resident"
                    )
                continue

            plans: list[tuple[torch.fx.Node, _Selector]] = []
            failure: tuple[torch.fx.Node, str] | None = None
            for user in narrowing:
                user_info = graph_infos.get(user.graph)
                if user_info is None:
                    failure = (user, "it is not in a graph being compiled")
                    break
                selector = _normalize(
                    user,
                    user_info,
                    parents,
                    graph_infos,
                    config,
                    block,
                    placeholder_to_outer,
                )
                if isinstance(selector, str):
                    failure = (user, selector)
                    break
                plans.append((user, selector))

            if failure is not None:
                culprit, reason = failure
                culprit.meta[_REJECTION] = reason
                for user in narrowing:
                    user.meta.setdefault(
                        _REJECTION,
                        "another run out of the same block could not be planned "
                        f"at {_where(culprit)}, so the block cannot stay resident",
                    )
                continue

            node.meta[_LOAD_AS_REF] = True
            for user, selector in plans:
                user.meta[_SUBVIEW] = _Subview(node, selector)

    _reject_unplanned(graphs)


def _reject_unplanned(graphs: list[GraphInfo]) -> None:
    """Fail on every narrowing subscript that planning did not cover.

    There is no materializing lowering for a run: Mosaic has neither
    ``dynamic_slice`` nor a gather, so a run selected out of a value in vector
    registers cannot be expressed at all.
    """
    for info in graphs:
        for node in info.graph.nodes:
            if (
                node.op != "call_function"
                or node.target is not subscript
                or not narrowing_positions(node)
                or subview_of(node) is not None
            ):
                continue
            reason = node.meta.get(_REJECTION) or "it could not be planned"
            raise exc.InvalidIndexingType(
                f"hl.subscript at {_where(node)} selects part of a dimension, "
                f"which the Pallas backend can only read out of a resident "
                f"block, but {reason}."
            )


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def _mask_expr(
    state: CodegenState, block_id: int, dim: int, result: ast.AST
) -> ast.AST:
    """Zero the positions a tile-driven read takes past its loop's extent."""
    mask_var = state.codegen.mask_var(block_id)
    if mask_var is None:
        return result
    assert state.fx_node is not None
    value = state.fx_node.meta["val"]
    assert isinstance(value, torch.Tensor)
    dtype = CompileEnvironment.current().backend.dtype_str(value.dtype)
    expand = state.tile_strategy.expand_str([*value.shape], dim)
    return expr_from_string(
        f"{{result}} * ({mask_var}.astype({dtype}){expand})", result=result
    )


def _codegen_subview_read(state: CodegenState, plan: _Subview) -> ast.AST:
    """Read one contiguous run out of a block emitted as a Pallas Ref."""
    selector = plan.selector
    assert state.fx_node is not None
    rank = _input_rank(state.fx_node)
    width = selector.width
    if selector.kind is _SelectorKind.TILE:
        resolved = state.device_function.resolved_block_size(selector.block_id)
        assert isinstance(resolved, int)
        begin, width = state.codegen.offset_var(selector.block_id), resolved
    elif selector.kind is _SelectorKind.DYNAMIC:
        start = state.device_function.literal_expr(selector.dynamic_begin)
        begin = f"jnp.clip({start}, 0, {selector.block_extent - width})"
    else:
        begin = str(selector.begin)

    parts = [":"] * rank
    parts[selector.dim] = f"pl.ds({begin}, {width})"
    result = expr_from_string(f"{{base}}[{', '.join(parts)}]", base=state.ast_arg(0))
    if selector.squeeze:
        keys = ["0" if dim == selector.dim else ":" for dim in range(rank)]
        result = expr_from_string(f"{{result}}[{', '.join(keys)}]", result=result)
    if selector.masked:
        result = _mask_expr(state, selector.block_id, selector.dim, result)
    return result


@_decorators.codegen(subscript, "pallas")
def _(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    plan = subview_of(state.fx_node)
    if plan is not None:
        return _codegen_subview_read(state, plan)
    # Planning rejected every narrowing subscript it could not cover, so what
    # is left only adds or keeps whole dimensions.
    # pyrefly: ignore [missing-attribute]
    return subscript._codegen["common"](state)


@_decorators.codegen(split, "pallas")
def _(state: CodegenState) -> list[ast.AST]:
    tensor = state.ast_arg(0)
    return [
        expr_from_string("{tensor}[..., 0]", tensor=tensor),
        expr_from_string("{tensor}[..., 1]", tensor=tensor),
    ]


@_decorators.codegen(join, "pallas")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "jnp.stack(jnp.broadcast_arrays({tensor0}, {tensor1}), axis=-1)",
        tensor0=state.ast_arg(0),
        tensor1=state.ast_arg(1),
    )
