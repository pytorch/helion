"""Pallas-backend codegen for ops defined in ``helion.language.view_ops``.

This module also implements *subview folding*.  When every consumer of a tile
load narrows one dimension to a contiguous run -- ``block = x[outer, :, :]``
followed by ``block[local.begin, :, :]`` inside a nested loop -- the load is
emitted as a Pallas Ref rather than a materialized array, and each consumer
becomes a small read out of that Ref.

The load node stays where it is, in the outer loop body, so the block is still
staged and DMA double-buffered once per outer tile; only the read of an
individual slice moves into the inner loop.  Folding is an optimization only:
``hl.subscript`` has a materializing lowering for every index form it accepts,
so declining to fold always leaves a working kernel.
"""

from __future__ import annotations

import ast
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
    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

# Set on a ``load`` node that subview folding wants emitted as a Pallas Ref.
_LOAD_AS_REF = "pallas_load_as_ref"
# Set on a ``subscript`` node that reads out of such a Ref.
_SUBVIEW = "pallas_subview"
# Set on a ``subscript`` node whose index is a node: the node that defines it,
# after following renames and loop captures, or None if it is not a node.
_INDEX_SOURCE = "pallas_index_source"

CONFIG_KEY = "pallas_resident_subviews"


@dataclasses.dataclass(frozen=True)
class _Block:
    """The tile of one dimension that a foldable load brings into VMEM."""

    dim: int
    """Dimension of the tensor the load tiles."""

    block_id: int
    """Tile loop that produced the block."""

    extent: int
    """Size of the block along ``dim``."""

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
    """One contiguous run narrowed out of a single dimension of a block."""

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

    squeeze: bool = False
    """The narrowed dimension does not appear in the result."""

    masked: bool = False
    """The run can extend past the rows the tile actually covers, so the read
    has to be multiplied by the driving loop's mask."""


@dataclasses.dataclass(frozen=True)
class _Subview:
    """A ``subscript`` that reads a contiguous run out of a folded block load."""

    source: torch.fx.Node
    selector: _Selector


def _node_value(value: object) -> object:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _input_rank(node: torch.fx.Node) -> int:
    """Rank of the value a ``subscript`` narrows, or -1 if it is not a tensor."""
    input_node = node.args[0]
    value = _node_value(input_node)
    return value.ndim if isinstance(value, torch.Tensor) else -1


def subview_of(node: torch.fx.Node) -> _Subview | None:
    """Return the folding plan recorded for a ``subscript`` node, if any."""
    plan = node.meta.get(_SUBVIEW)
    return plan if isinstance(plan, _Subview) else None


def load_is_ref(node: torch.fx.Node) -> bool:
    """True when subview folding wants this ``load`` emitted as a Pallas Ref."""
    return node.meta.get(_LOAD_AS_REF) is True


# ---------------------------------------------------------------------------
# Discovery
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
    name, which a Pallas Ref survives unchanged, so they are transparent here
    and need no annotation of their own.
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


def _resolve_index_source(
    node: object, placeholder_to_outer: dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node | None:
    """Follow renames and loop captures back to the node defining an index.

    An index written in one scope and used in another arrives as a chain of
    ``_new_var`` renames and capture placeholders.  Both lowerings need the
    node underneath to tell a tile run apart from an arbitrary gather.
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


def _record_index_sources(
    graphs: list[GraphInfo],
    captures: dict[torch.fx.Node, list[tuple[torch.fx.Node, torch.fx.Node]]],
) -> None:
    """Resolve every ``subscript``'s node-valued index once, for both lowerings."""
    placeholder_to_outer = {
        placeholder: outer
        for outer, edges in captures.items()
        for placeholder, _parent in edges
    }
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or node.target is not subscript:
                continue
            indices = node.args[1]
            if not isinstance(indices, (list, tuple)):
                continue
            for index in indices:
                if isinstance(index, torch.fx.Node):
                    node.meta[_INDEX_SOURCE] = _resolve_index_source(
                        index, placeholder_to_outer
                    )
                    break


def _index_source(node: torch.fx.Node) -> torch.fx.Node | None:
    source = node.meta.get(_INDEX_SOURCE)
    return source if isinstance(source, torch.fx.Node) else None


def _mutated_tensor_ids(graphs: list[GraphInfo]) -> set[int]:
    """Ids of fake tensors written by a store or atomic anywhere on device.

    A folded load reads its Ref at the consumer's program point instead of at
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


def _foldable_block(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parent: torch.fx.Node | None,
    config: Config,
    mutated: set[int],
) -> _Block | None:
    """Describe the block a foldable ``load`` brings in, or return None.

    A foldable load tiles exactly one dimension and passes every other one
    through, so its result keeps the tensor's rank and a ``pl.ds`` slice of the
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
        or tensor.ndim != value.ndim
        or node.args[2] is not None
        or not isinstance(indices, (list, tuple))
        or id(tensor) in mutated
        or any(
            isinstance(pattern, IndirectGatherPattern)
            for pattern in node.meta.get("indexing_patterns") or ()
        )
    ):
        return None

    tiled = [dim for dim, index in enumerate(indices) if index != slice(None)]
    if len(tiled) != 1:
        return None
    dim = tiled[0]

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(_node_value(indices[dim]))
    if block_id is None:
        return None
    extent = env.block_sizes[block_id].from_config(config)
    if not isinstance(extent, int):
        return None
    lane_extent = env.size_hint(value.shape[-1])
    if _slice_addressing(value, dim, lane_extent) is not SliceAddressing.DIRECT:
        return None

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


def _tile_driven_selector(
    index: torch.fx.Node,
    graph_info: GraphInfo,
    parent: torch.fx.Node | None,
    config: Config,
    block: _Block,
) -> _Selector | None:
    """Validate a ``tile.begin`` or ``tile.index`` narrowing of ``block``."""
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
            return None
        inner_block_id = origin.origin.block_id
    elif index.target is tile_index and index.args:
        resolved = env.resolve_block_id(_node_value(index.args[0]))
        if resolved is None:
            return None
        inner_block_id = resolved
    else:
        return None

    bounds = _loop_bounds(graph_info, parent, inner_block_id)
    if bounds is None:
        return None
    begin, end = bounds
    if not isinstance(begin, (int, torch.SymInt)) or not env.known_equal(begin, 0):
        return None

    width = env.block_sizes[inner_block_id].from_config(config)
    if not isinstance(width, int) or width < 1 or width > block.extent:
        return None
    if squeeze and width != 1:
        return None
    # The tail iteration must stop at the end of the block, not past it.
    if block.extent % width != 0:
        return None

    if isinstance(end, int):
        # A constant trip count only stays within live rows when the whole
        # block is live and the loop covers exactly it.
        if not block.whole_dim or end != block.extent:
            return None
    elif isinstance(end, torch.SymInt):
        if not block.whole_dim and not _bounded_by_block(end, block.block_id):
            return None
    else:
        return None

    return _Selector(
        _SelectorKind.TILE,
        block.dim,
        width,
        block.extent,
        block_id=inner_block_id,
        squeeze=squeeze,
        masked=not squeeze,
    )


def _iota_selector(index: torch.fx.Node, block: _Block) -> _Selector | None:
    """Validate a constant-width run at a data-dependent start.

    ``block[live - 3 : live]`` traces to an ``iota`` of a constant length, so
    the width is static even though the start is not.  The start is clamped
    into the block at codegen, which is what ``jax.lax.dynamic_slice`` does
    with an out-of-range start, so the folded and materializing lowerings
    agree on every input.
    """
    if index.target is not torch.ops.prims.iota.default or not index.args:
        return None
    width = _node_value(index.args[0])
    start = _node_value(index.kwargs.get("start"))
    if (
        not isinstance(width, int)
        or width < 1
        or width > block.extent
        or index.kwargs.get("step") != 1
        or not isinstance(start, (int, torch.SymInt))
    ):
        return None
    # A start past the live rows would read stale data rather than the zeros
    # the outer load's mask would have produced, so require a full block.
    if not block.whole_dim:
        return None
    return _Selector(_SelectorKind.DYNAMIC, block.dim, width, block.extent)


def _selector(
    node: torch.fx.Node,
    graph_info: GraphInfo,
    parent: torch.fx.Node | None,
    config: Config,
    block: _Block,
) -> _Selector | None:
    """Validate one ``subscript`` of a folded block, or return None."""
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or any(
        index is None for index in indices
    ):
        return None
    narrowed = [
        position for position, index in enumerate(indices) if index != slice(None)
    ]
    # With no ``None`` entries the subscript positions are the block's
    # dimensions, so the narrowed position has to be the tiled one.
    if len(narrowed) != 1 or narrowed[0] != block.dim:
        return None
    index = indices[block.dim]

    if isinstance(index, int):
        if index < 0 or index >= block.extent or not block.whole_dim:
            return None
        return _Selector(
            _SelectorKind.CONSTANT,
            block.dim,
            1,
            block.extent,
            begin=index,
            squeeze=True,
        )

    if isinstance(index, slice):
        if (
            index.step not in (None, 1)
            or not isinstance(index.start, int)
            or not isinstance(index.stop, int)
            or index.start < 0
            or index.stop <= index.start
            or index.stop > block.extent
            or not block.whole_dim
        ):
            return None
        return _Selector(
            _SelectorKind.CONSTANT,
            block.dim,
            index.stop - index.start,
            block.extent,
            begin=index.start,
        )

    if isinstance(index, torch.fx.Node):
        # Use the resolved definition, so an index that crossed a loop or
        # branch boundary still folds.
        source = _index_source(node) or index
        if source.target is torch.ops.prims.iota.default:
            return _iota_selector(source, block)
        return _tile_driven_selector(source, graph_info, parent, config, block)
    return None


def plan_subview_folding(graphs: list[GraphInfo], config: Config) -> None:
    """Mark block loads that can stay Refs and the subviews that read them."""
    for info in graphs:
        for node in info.graph.nodes:
            node.meta.pop(_LOAD_AS_REF, None)
            node.meta.pop(_SUBVIEW, None)
            node.meta.pop(_INDEX_SOURCE, None)

    parents = _control_flow_parents(graphs)
    captures = _capture_map(graphs, parents)
    # Recorded even when folding is off: the materializing lowering needs it
    # too, to tell a tile run apart from an arbitrary gather.
    _record_index_sources(graphs, captures)

    enabled = config.get(CONFIG_KEY, True)
    if not isinstance(enabled, bool):
        raise exc.InvalidConfig(f"{CONFIG_KEY} must be True or False, got {enabled!r}.")
    if not enabled:
        return

    graph_infos = {info.graph: info for info in graphs}
    mutated = _mutated_tensor_ids(graphs)

    def parent_of(info: GraphInfo) -> torch.fx.Node | None:
        entry = parents.get(info.graph_id)
        return entry[0] if entry is not None else None

    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or node.target is not load:
                continue
            block = _foldable_block(node, info, parent_of(info), config, mutated)
            if block is None:
                continue

            rank = node.meta["val"].ndim
            plans: list[tuple[torch.fx.Node, _Selector]] = []
            for user in _effective_users(node, captures):
                user_info = graph_infos.get(user.graph)
                if (
                    user.op != "call_function"
                    or user.target is not subscript
                    or user_info is None
                    or _input_rank(user) != rank
                ):
                    plans = []
                    break
                selector = _selector(
                    user, user_info, parent_of(user_info), config, block
                )
                if selector is None:
                    plans = []
                    break
                plans.append((user, selector))

            if not plans:
                continue
            node.meta[_LOAD_AS_REF] = True
            for user, selector in plans:
                user.meta[_SUBVIEW] = _Subview(node, selector)


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


def _dynamic_begin_expr(state: CodegenState, selector: _Selector) -> str:
    """Clamped start of a data-dependent run, matching ``lax.dynamic_slice``."""
    assert state.fx_node is not None
    indices = state.fx_node.args[1]
    assert isinstance(indices, (list, tuple))
    iota = indices[selector.dim]
    assert isinstance(iota, torch.fx.Node)
    start = state.device_function.literal_expr(_node_value(iota.kwargs.get("start")))
    return f"jnp.clip({start}, 0, {selector.block_extent - selector.width})"


def _codegen_subview_read(state: CodegenState, plan: _Subview) -> ast.AST:
    """Read one contiguous run out of a block that was emitted as a Ref."""
    selector = plan.selector
    assert state.fx_node is not None
    rank = _input_rank(state.fx_node)
    width = selector.width
    if selector.kind is _SelectorKind.TILE:
        resolved = state.device_function.resolved_block_size(selector.block_id)
        assert isinstance(resolved, int)
        begin, width = state.codegen.offset_var(selector.block_id), resolved
    elif selector.kind is _SelectorKind.DYNAMIC:
        begin = _dynamic_begin_expr(state, selector)
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


def _contiguous_narrowing(
    state: CodegenState, position: int
) -> tuple[ast.AST, int, bool, int] | None:
    """Return ``(offset, width, squeeze, block_id)`` for a contiguous run.

    ``tile.begin`` selects a single row and drops the dimension; ``tile.index``
    selects a whole inner block and keeps it, masked at the tail; an ``iota``
    selects a constant-width run at a data-dependent start.  Returns None for
    an index this cannot prove contiguous, which the caller gathers instead.
    """
    assert state.fx_node is not None
    ast_args = state.ast_args[1]
    assert isinstance(ast_args, (list, tuple))
    proxy = state.proxy_arg(1)
    assert isinstance(proxy, (list, tuple))

    if isinstance(proxy[position], torch.SymInt):
        offset = ast_args[position]
        assert isinstance(offset, ast.AST)
        return offset, 1, True, -1

    source = _index_source(state.fx_node)
    if source is None:
        return None
    env = CompileEnvironment.current()
    if source.target is tile_index and source.args:
        block_id = env.resolve_block_id(_node_value(source.args[0]))
        width = (
            state.device_function.resolved_block_size(block_id)
            if block_id is not None
            else None
        )
        if block_id is not None and isinstance(width, int):
            return (
                expr_from_string(state.codegen.offset_var(block_id)),
                width,
                False,
                block_id,
            )
    elif source.target is torch.ops.prims.iota.default and source.args:
        width = _node_value(source.args[0])
        start = _node_value(source.kwargs.get("start"))
        if isinstance(width, int) and isinstance(start, (int, torch.SymInt)):
            offset = expr_from_string(state.device_function.literal_expr(start))
            return offset, width, False, -1
    return None


def _axis_extent(state: CodegenState, axis: int) -> int | None:
    """Resolved size of the indexed value along ``axis`` for this config."""
    assert state.fx_node is not None
    value = _node_value(state.fx_node.args[0])
    if not isinstance(value, torch.Tensor):
        return None
    size = value.shape[axis]
    if isinstance(size, int):
        return size
    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(size)
    if block_id is not None:
        resolved = state.device_function.resolved_block_size(block_id)
        return resolved if isinstance(resolved, int) else None
    concrete = env.try_concretize_symint(size)
    return concrete if isinstance(concrete, int) else None


def _codegen_value_subscript(state: CodegenState) -> ast.AST:
    """Index a materialized array.

    Covers every form ``hl.subscript``'s fake accepts: any number of ``None``
    and ``:`` entries plus at most one narrowing entry, which may be a constant
    index, a constant slice, a ``tile.begin`` offset, or a ``tile.index`` run.
    """
    index = list(state.proxy_arg(1))  # pyrefly: ignore [bad-argument-type]
    base = state.ast_arg(0)

    def is_full_slice(value: object) -> bool:
        return isinstance(value, slice) and (value.start, value.stop, value.step) == (
            None,
            None,
            None,
        )

    dynamic = [
        position
        for position, value in enumerate(index)
        if isinstance(value, (torch.SymInt, torch.Tensor))
    ]
    if len(dynamic) > 1:
        raise exc.InvalidIndexingType(repr(index[dynamic[1]]))

    # ``None`` adds a dimension, so an entry's axis in the input value is its
    # position among the non-``None`` entries.
    axis_of: list[int] = []
    axis = 0
    for value in index:
        axis_of.append(-1 if value is None else axis)
        axis += 0 if value is None else 1

    squeeze_position = -1
    mask: tuple[int, int] | None = None
    if dynamic:
        position = dynamic[0]
        run = _contiguous_narrowing(state, position)
        if run is None:
            # An index this cannot prove contiguous -- an offset tile run, or
            # any other computed row vector -- is an ordinary gather.
            index_ast = state.ast_args[1]
            assert isinstance(index_ast, (list, tuple))
            rows = index_ast[position]
            assert isinstance(rows, ast.AST)
            base = expr_from_string(
                f"jnp.take({{base}}, {{rows}}, axis={axis_of[position]})",
                base=base,
                rows=rows,
            )
        else:
            offset, width, squeeze, block_id = run
            base = expr_from_string(
                f"jax.lax.dynamic_slice_in_dim({{base}}, {{offset}}, {width}, "
                f"{axis_of[position]})",
                base=base,
                offset=offset,
            )
            if squeeze:
                squeeze_position = position
            elif block_id >= 0:
                mask = (block_id, axis_of[position])

    keys: list[str] = []
    for position, value in enumerate(index):
        if value is None:
            keys.append("None")
        elif position == squeeze_position:
            keys.append("0")
        elif position in dynamic or is_full_slice(value):
            keys.append(":")
        elif isinstance(value, slice):
            # Python clips a slice to the dimension but the traced shape does
            # not, so a run reaching past the end would give the value a
            # different shape than the rest of the graph was built for.
            extent = _axis_extent(state, axis_of[position])
            if (
                isinstance(value.stop, int)
                and extent is not None
                and value.stop > extent
            ):
                raise exc.InvalidConfig(
                    f"hl.subscript slice {value.start}:{value.stop} reaches past "
                    f"the {extent}-element dimension it narrows."
                )
            if value.step not in (None, 1):
                raise exc.InvalidIndexingType(repr(value))
            start = "" if value.start is None else str(value.start)
            stop = "" if value.stop is None else str(value.stop)
            keys.append(f"{start}:{stop}")
        elif isinstance(value, int):
            keys.append(str(value))
        else:
            raise exc.InvalidIndexingType(repr(value))

    result = expr_from_string(f"{{base}}[{', '.join(keys)}]", base=base)
    if mask is not None:
        result = _mask_expr(state, mask[0], mask[1], result)
    return result


@_decorators.codegen(subscript, "pallas")
def _(state: CodegenState) -> ast.AST:
    assert state.fx_node is not None
    plan = subview_of(state.fx_node)
    if plan is not None and state.device_function.pallas_ref_loads.get(plan.source):
        return _codegen_subview_read(state, plan)
    return _codegen_value_subscript(state)


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
