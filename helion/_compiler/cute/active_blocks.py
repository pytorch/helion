"""CuTe uses of the logical-axis liveness analysis (``helion._compiler.active_blocks``)."""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch
from torch.fx.node import Node
from torch.utils._pytree import tree_leaves

from ... import exc
from ...language import _tracing_ops
from ..active_blocks import full_slice_block_ids
from ..compile_environment import CompileEnvironment
from ..device_ir import IfGraphInfo
from ..device_ir import control_flow_child_graph_ids
from ..host_function import HostFunction
from ..node_masking import METADATA_TARGETS
from ..node_masking import if_result_source
from ..node_masking import loop_carries
from .loop_nesting import graph_tile_loop_paths

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState
    from .loop_nesting import TileLoopPath


def match_active_reduction_block_id(
    state: CodegenState, node: Node, size: object
) -> int | None:
    """The one active tile block whose extent is ``size``, for a reduction over
    a dim that is not a reduction block (``Backend.reduction_active_block_id``):
    the reduction then runs as a block-level strategy over that tile's threads.
    ``None`` when no block or more than one block matches."""
    env = CompileEnvironment.current()
    candidates: set[int] = set()
    block_id = env.resolve_block_id(size)
    if block_id is not None:
        block_id = env.resolve_codegen_block_id(
            block_id,
            state.codegen,
            node.graph,
        )
        if state.codegen.active_device_loops.get(block_id) or (
            state.codegen.device_function.tile_strategy.thread_axis_for_block_id(
                block_id
            )
            is not None
        ):
            candidates.add(block_id)
    for strategy in state.codegen.device_function.tile_strategy.strategies:
        for candidate_block_id in strategy.block_ids:
            if (
                state.codegen.device_function.tile_strategy.thread_axis_for_block_id(
                    candidate_block_id
                )
                is None
            ):
                continue
            candidate_size = env.block_sizes[candidate_block_id].size
            candidate_source = getattr(
                env.block_sizes[candidate_block_id].block_size_source,
                "value",
                None,
            )
            if (
                isinstance(size, torch.SymInt)
                and isinstance(candidate_source, torch.SymInt)
                and candidate_source._sympy_() == size._sympy_()
            ):
                candidates.add(candidate_block_id)
                continue
            if isinstance(size, (int, torch.SymInt)) and isinstance(
                candidate_size, (int, torch.SymInt)
            ):
                if env.known_equal(candidate_size, size):
                    candidates.add(candidate_block_id)

    seen: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            key = id(loop_state)
            if key in seen:
                continue
            seen.add(key)
            for candidate_block_id, info in loop_state.block_id_to_info.items():
                if isinstance(size, (int, torch.SymInt)) and info.is_end_matching(size):
                    candidates.add(candidate_block_id)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def reject_tile_alias_reductions(
    fn: DeviceFunction,
    block_ids: Iterable[int],
    live_graph_ids: frozenset[int] | None,
) -> None:
    """Refuse a live reduction block that reuses the variable of a block a
    tile loop iterates, where the two meet.

    A full slice of a host tensor dim sized by a registered block size
    (``buf = x.new_empty([m, block_n])`` read as ``buf[tile_m, :]``)
    allocates a reduction block that reuses that block's variable
    (``CompileEnvironment.allocate_reduction_dimension``).  When a loop tiles
    the same block (``hl.tile(n, block_size=block_n)``), the slice's dim and
    the tile's are one symbol: shape propagation pairs their elements by
    position, as Triton's ``tl.arange`` lanes do, but CuTe gives each block
    coordinates of its own.  Beside a grid tile the slice read the tile's
    index, past the buffer's first ``block_n`` columns; beside a device-loop
    tile it got lanes of its own and each lane stored a different column into
    the same element, also when the row was read before the loop.  The kernel
    is refused where the two meet (``_alias_meets_tile``); elsewhere the tile
    has no coordinates and the slice keeps its own, as zeroing the buffer
    before the loop does.  ``block_ids`` are the reduction blocks that would
    get a strategy, and ``live_graph_ids`` the graphs this config emits
    (``Backend.constexpr_live_graph_ids``: the untaken side of a constant
    branch is never emitted, so it meets nothing; ``None`` when every graph
    is).  Indexing the dim with ``tile.index - tile.begin`` inside the loop
    allocates no reduction block.
    """
    env = CompileEnvironment.current()
    aliases: dict[int, list[int]] = {}
    for block_id in block_ids:
        canonical = env.canonical_block_id(block_id)
        if canonical != block_id:
            aliases.setdefault(canonical, []).append(block_id)
    if not aliases:
        return
    paths = [
        (info, path)
        for info, path in graph_tile_loop_paths(
            HostFunction.current().device_ir, fn.codegen.codegen_graphs
        )
        if live_graph_ids is None or info.graph_id in live_graph_ids
    ]
    for canonical, alias_ids in aliases.items():
        meeting = _alias_meets_tile(fn, canonical, paths, live_graph_ids)
        if meeting is not None:
            hint = (
                "index the dim with tile.index - tile.begin"
                if meeting == _ACCESSED_INSIDE
                else "access the buffer inside the loop and index the dim with "
                "tile.index - tile.begin"
            )
            raise exc.BackendUnsupported(
                "cute",
                "a full slice of a host tensor dim sized by a tile loop's block "
                f"size {meeting} (reduction block {alias_ids[0]} reuses tile block "
                f"{canonical}'s variable, and CuTe has no lanes that pair its "
                f"elements with the tile's); {hint} instead",
            )


_ACCESS = "access"
_LOOP = "loop"
_ACCESSED_INSIDE = "is accessed inside that loop"


class _Flow(NamedTuple):
    """Where the values of a graph outside the tile's loop come from
    (``_alias_meets_tile``): an access of the slice (``_ACCESS``), the loop
    (``_LOOP``) or the graph's inputs (placeholder indices)."""

    outputs: tuple[frozenset[int | str], ...]
    """The sources of each output."""
    into_access: frozenset[int | str]
    """The sources of the values an access reads (the row it stores), or
    that decide whether it runs (the test of a tensor ``if`` or the
    condition of a ``while`` around it)."""
    into_loop: frozenset[int | str]
    """The sources of the values the loop reads, or that decide whether it
    runs."""
    joins: frozenset[frozenset[int | str]]
    """The sources of the values each operation combines, which pair the
    elements of their shared dim by position (the loop's row times the
    slice's, after the loop)."""
    accesses: bool
    """Whether the graph, or one it runs, accesses the slice."""
    loops: bool
    """Whether the graph, or one it runs, runs the loop."""


def _if_branches(node: Node) -> list[tuple[int, list[object]]]:
    """Each branch of the ``_if`` ``node``, the if branch first: its graph
    and the arguments it receives."""
    if_id, else_id = control_flow_child_graph_ids(node)
    return [
        (if_id, cast("list[object]", node.args[3])),
        (else_id, cast("list[object]", node.args[4])),
    ]


def _substitute(
    sources: frozenset[int | str], args: Sequence[frozenset[int | str]]
) -> frozenset[int | str]:
    """``sources`` in a graph, with each of its inputs replaced by the sources
    of the argument its caller passes (``args``)."""
    return frozenset(
        {source for source in sources if isinstance(source, str)}.union(
            *(args[source] for source in sources if isinstance(source, int))
        )
    )


def _alias_meets_tile(
    fn: DeviceFunction,
    tile: int,
    paths: Sequence[tuple[GraphInfo, TileLoopPath]],
    live_graph_ids: frozenset[int] | None,
) -> str | None:
    """How a full slice of a host tensor dim sized by ``tile``'s block meets a
    loop over ``tile``, for the refusal's message, or ``None``
    (``reject_tile_alias_reductions``); ``paths`` are the live codegen graphs
    with their tile loops (``graph_tile_loop_paths``), ``live_graph_ids`` the
    graphs the config emits (``None``: all).

    A slice meets the loop when it is accessed inside it (anywhere beside a
    grid tile), or when a value carrying their shared dim connects the access
    with the loop: a value the access reads flows into the loop, one the
    loop produces flows into the access (a row accumulated over the tile and
    stored into the buffer), or an operation outside the loop reads values
    of both (the loop's row times the slice's).  The test of a tensor ``if``
    or a ``while`` condition counts as a value read by the access or the
    loop it runs, and by each of its results, since it decides element by
    element whether that runs and which value each element of a result
    takes (zeroing the buffer under a mask the loop computes, or storing a
    row the mask picks).  A value whose shape lost the dim (a sum over the
    slice) carries nothing across; nor does memory (the loop's atomics into
    the buffer, read back after it), nor a merge of two control-flow paths
    (``_phi``) that no operation reads.  Values flow into nested graphs
    through their inputs and out through the results their outputs make: a
    branch's result carries what its branch computes it from, not everything
    the branch touches (a value it builds beside zeroing the buffer), and a
    loop's results, ``for`` or ``while``, also what its body carries from one
    iteration to the next.  The tile's loop reads an input where its body
    reads the input's value, not where the body overwrites it, reading only
    its shape, but also where a branch that does not overwrite it passes it
    through.  The untaken side of a constant branch takes no part, not even
    through a value it passes through unchanged.
    """
    env = CompileEnvironment.current()
    graphs = fn.codegen.codegen_graphs

    def on_tile(size: object) -> bool:
        # Through a product too: a row flattened to [tile_m * block_n] and
        # reshaped back still carries the dim.
        return isinstance(size, torch.SymInt) and any(
            (block_id := env.get_block_id(symbol)) is not None
            and env.canonical_block_id(block_id) == tile
            for symbol in size._sympy_().free_symbols
        )

    def carries_tile(node: Node) -> bool:
        return any(
            on_tile(size)
            for value in tree_leaves(node.meta.get("val"))
            if isinstance(value, torch.Tensor)
            for size in value.shape
        )

    def slices_tile(node: Node) -> bool:
        return tile in full_slice_block_ids(node)

    def live(graph_id: int) -> bool:
        return live_graph_ids is None or graph_id in live_graph_ids

    def result_side(node: Node, index: int) -> int:
        """The branch whose run makes result ``index`` of the ``_if``
        ``node``, also where it passes a value through."""
        info = graphs[cast("int", node.args[1])]
        assert isinstance(info, IfGraphInfo)
        assert info.branches_outputs is not None
        return cast("int", node.args[1 + index // len(info.branches_outputs)])

    def reads_elements(value: Node) -> bool:
        """Whether ``value``'s elements are read where it is used, not only
        its shape: also through a copy (``_new_var``) and the branches of an
        ``if`` it is passed into, which read it where they read the input it
        becomes, or pass that through, as the side that does not write it
        does (the writing branch takes it as an input, reading only its
        shape where it overwrites it)."""
        for user in value.users:
            if user.target in METADATA_TARGETS:
                continue
            if user.target is _tracing_ops._new_var:
                if reads_elements(user):
                    return True
                continue
            if user.target is not _tracing_ops._if or user.args[0] is value:
                return True
            for child, args in _if_branches(user):
                if not live(child):
                    continue
                graph = graphs[child].graph
                (output,) = graph.find_nodes(op="output")
                passed = tree_leaves(output.args[0])
                for arg, placeholder in zip(
                    args, graph.find_nodes(op="placeholder"), strict=True
                ):
                    if arg is value and (
                        placeholder in passed or reads_elements(placeholder)
                    ):
                        return True
            for result in user.users:
                if (
                    result.target is operator.getitem
                    and live(result_side(user, cast("int", result.args[1])))
                    and if_result_source(user, cast("int", result.args[1]), graphs)
                    is value
                    and reads_elements(result)
                ):
                    return True
        return False

    inside = {
        info.graph_id
        for info, path in paths
        if any(
            env.canonical_block_id(block_id) == tile
            for loop_block_ids in path
            for block_id in loop_block_ids
        )
    }
    if not inside:
        return None
    if any(
        slices_tile(node)
        for info, _path in paths
        if info.graph_id in inside
        for node in info.graph.nodes
    ):
        return _ACCESSED_INSIDE

    @functools.cache
    def flow(graph_id: int) -> _Flow:
        """The flow of a graph outside the loop, by its inputs."""
        graph = graphs[graph_id].graph
        sources: dict[Node, frozenset[int | str]] = {
            placeholder: frozenset([index])
            for index, placeholder in enumerate(graph.find_nodes(op="placeholder"))
        }
        results: dict[Node, dict[int, frozenset[int | str]]] = {}
        into_access: set[int | str] = set()
        into_loop: set[int | str] = set()
        joins: set[frozenset[int | str]] = set()
        accesses = loops = False

        def of(value: object) -> frozenset[int | str]:
            return (
                sources.get(value, frozenset())
                if isinstance(value, Node)
                else frozenset()
            )

        def call(child: _Flow, args: Sequence[frozenset[int | str]]) -> None:
            nonlocal accesses, loops
            into_access.update(_substitute(child.into_access, args))
            into_loop.update(_substitute(child.into_loop, args))
            joins.update(_substitute(join, args) for join in child.joins)
            accesses |= child.accesses
            loops |= child.loops

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            read = frozenset().union(*map(of, node.all_input_nodes))
            children = control_flow_child_graph_ids(node)
            if not children:
                if node.target is operator.getitem and node.args[0] in results:
                    read = results[cast("Node", node.args[0])].get(
                        cast("int", node.args[1]), frozenset()
                    )
                if slices_tile(node):
                    into_access.update(read)
                    read |= {_ACCESS}
                    accesses = True
                # An operation pairs the elements of what it reads; a merge
                # of two control-flow paths (``_phi``), a copy or a result
                # passes its sources on to the operations that read it.
                if len(read) > 1 and node.target not in (
                    _tracing_ops._phi,
                    _tracing_ops._new_var,
                    operator.getitem,
                ):
                    joins.add(read)
                if read and carries_tile(node):
                    sources[node] = read
                continue
            indices = [
                user.args[1]
                for user in node.users
                if user.target is operator.getitem and isinstance(user.args[1], int)
            ]
            if any(child in inside for child in children):
                # The loop over the tile: an input reaches it where its body
                # reads its value, not where the body overwrites it and reads
                # only its shape.
                if _tracing_ops.is_for_loop_target(node.target):
                    loop_args = cast("list[object]", node.args[3])
                    placeholders = graphs[children[0]].graph.find_nodes(
                        op="placeholder"
                    )
                    read = frozenset().union(
                        *(
                            of(arg)
                            for arg, placeholder in zip(
                                loop_args, placeholders, strict=True
                            )
                            if reads_elements(placeholder)
                        ),
                        *(
                            of(value)
                            for value in node.all_input_nodes
                            if value not in loop_args
                        ),
                    )
                into_loop.update(read)
                loops = True
                results[node] = dict.fromkeys(indices, read | {_LOOP})
            elif node.target is _tracing_ops._if:
                branches = {
                    child: [*map(of, args)]
                    for child, args in _if_branches(node)
                    if live(child)
                }
                for child, args in branches.items():
                    call(flow(child), args)
                # A tensor test decides element by element whether a branch's
                # access or loop runs, and which side's value each element
                # of a result takes.
                info = graphs[cast("int", node.args[1])]
                assert isinstance(info, IfGraphInfo)
                test = of(node.args[0]) if info.predicate_is_tensor else frozenset()
                if any(flow(child).accesses for child in branches):
                    into_access.update(test)
                if any(flow(child).loops for child in branches):
                    into_loop.update(test)
                results[node] = {}
                for index in indices:
                    # None on the untaken side, also where it passes a value
                    # through.
                    if result_side(node, index) not in branches:
                        continue
                    origin = if_result_source(node, index, graphs)
                    if not isinstance(origin, tuple):
                        results[node][index] = of(origin) | test
                    else:
                        child, output = origin
                        results[node][index] = test | _substitute(
                            flow(child).outputs[output], branches[child]
                        )
            else:
                # A for loop, or a while loop (its condition then its body).
                is_while = node.target is _tracing_ops._while_loop
                assert is_while or _tracing_ops.is_for_loop_target(node.target)
                runs = cast(
                    "tuple[int, ...]", node.args[:2] if is_while else node.args[:1]
                )
                body = runs[-1]
                args = [*map(of, cast("list", node.args[2 if is_while else 3]))]
                placeholders = graphs[body].graph.find_nodes(op="placeholder")
                carries = loop_carries(node, graphs[body].graph)
                while True:
                    outputs = [
                        _substitute(output, args) for output in flow(body).outputs
                    ]
                    carried = {
                        placeholders.index(carry.placeholder): outputs[carry.index]
                        for carry in carries
                    }
                    if all(value <= args[index] for index, value in carried.items()):
                        break
                    for index, value in carried.items():
                        args[index] |= value
                for child in runs:
                    call(flow(child), args)
                # A while condition decides element by element, as a tensor
                # test does, whether the body's access or loop runs and which
                # iteration's value each element of a result keeps.
                test = frozenset[int | str]()
                if is_while and (condition := flow(runs[0]).outputs):
                    test = _substitute(condition[0], args)
                if flow(body).accesses:
                    into_access.update(test)
                if flow(body).loops:
                    into_loop.update(test)
                results[node] = {index: outputs[index] | test for index in indices}
        (output,) = graph.find_nodes(op="output")
        values = output.args[0]
        return _Flow(
            tuple(map(of, values)) if isinstance(values, (list, tuple)) else (),
            frozenset(into_access),
            frozenset(into_loop),
            frozenset(joins),
            accesses,
            loops,
        )

    for root_id in HostFunction.current().device_ir.root_ids:
        if root_id in inside or not live(root_id):
            continue
        root = flow(root_id)
        if (
            _ACCESS in root.into_loop
            or _LOOP in root.into_access
            or any({_ACCESS, _LOOP} <= join for join in root.joins)
        ):
            return "meets that loop through a value carrying their shared dim"
    return None
