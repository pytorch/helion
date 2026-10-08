"""Loop carries whose dim is the loop's own lane-looped block.

The SIMT lowering holds a tile value at one element per thread and lane
iteration, but a value carried across the iterations of a device loop is one
scalar per thread.  When the loop's own block has more elements than
threads (a lane loop, or vector lanes), a carry with a dim on that block,
``acc = hl.zeros([tile_m, bn])`` updated in ``for tile_n in hl.tile(n,
block_size=bn)``, folds every lane of a thread into that one scalar.

Only an additive fold is still exact: from zero, ``acc = acc + x`` per lane
(or ``- x``, also in the branches of a runtime ``if``) leaves the sum of the
thread's lanes, and a sum over the block's dim after the loop
(``acc.sum(-1)``, the manual looped reduction) adds those partial sums
across the threads.  Any other update (``torch.maximum``, a rescale),
nonzero initial value, read of the carry inside the loop, or consumer after
the loop other than such a sum (``amax``, ``acc * acc``, a store) would see
the folded lanes, so ``check_lane_merged_carries`` refuses it.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx.node import Node

from ... import exc
from ...language._tracing_ops import _if
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _phi
from ...language.creation_ops import full
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ..device_ir import ForLoopGraphInfo
    from ..inductor_lowering import CodegenState

# Reads of a value's metadata, not of its elements.
_METADATA_TARGETS = (torch.ops.aten.sym_size.int, torch.ops.aten.sym_stride.int)
_ADDITIVE_TARGETS = (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor)
_SUM_TARGETS = (torch.ops.aten.sum.dim_IntList,)


def check_lane_merged_carries(state: CodegenState, info: ForLoopGraphInfo) -> None:
    """Refuse a carry of ``info``'s loop that folds the lanes of the loop's
    own block, unless it is a zero-initialized additive update consumed only
    by sums over that block's dim (see the module docstring)."""
    loop = state.fx_node
    if loop is None:
        return
    env = CompileEnvironment.current()
    device_function = state.device_function
    tile_strategy = device_function.tile_strategy
    lane_looped: set[int] = set()
    for block_id in info.block_ids:
        size = device_function.resolved_block_size(block_id)
        extent = tile_strategy.thread_extent_for_block_id(block_id)
        if isinstance(size, int) and size > 1 and (extent is None or extent < size):
            lane_looped.add(env.canonical_block_id(block_id))
    if not lane_looped:
        return
    inits = loop.args[3]
    assert isinstance(inits, (list, tuple))
    placeholders = info.graph.find_nodes(op="placeholder")
    (output,) = info.graph.find_nodes(op="output")
    outputs = output.args[0]
    assert isinstance(outputs, (list, tuple))
    for result in loop.users:
        if result.target is not operator.getitem:
            continue
        index = cast("int", result.args[1])
        for phi in result.users:
            if phi.target is not _phi or phi.args[1] is not result:
                continue
            init = phi.args[0]
            if init not in inits:
                continue
            placeholder = placeholders[inits.index(init)]
            value = placeholder.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            dims = [
                dim
                for dim, size in enumerate(value.shape)
                if (block_id := env.get_block_id(size)) is not None
                and env.canonical_block_id(block_id) in lane_looped
            ]
            if not dims:
                continue
            reason = _unfolded_use(
                state, init, placeholder, outputs[index], result, dims
            )
            if reason is not None:
                raise exc.BackendUnsupported(
                    "cute",
                    f"loop carry {phi.name} (shape {list(value.shape)}) has dim "
                    f"{dims[0]} on its own loop's block, whose threads each "
                    "hold several of its elements in one carried scalar; only "
                    "a zero-initialized sum (acc = acc + x, then acc.sum over "
                    f"that dim) survives that, but {reason}",
                )


def _unfolded_use(
    state: CodegenState,
    init: object,
    placeholder: Node,
    update: object,
    result: Node,
    dims: list[int],
) -> str | None:
    """Why the folded lanes of the carry would be observed, or None."""
    if not _is_zero(_skip_copies(init)):
        return "its initial value is not zero"
    if not isinstance(update, Node) or any(
        user.op != "output" for user in _element_users(update)
    ):
        return "the loop reads it other than to add to it"
    reason = _additive(state, update, placeholder)
    if reason is not None:
        return reason
    for phi in result.users:
        if phi.target is not _phi:
            return f"{phi.target} reads the loop's result"
        for user in _element_users(phi):
            if user.target not in _SUM_TARGETS or not _sums_dims(user, dims):
                return f"{user.target} reads it after the loop"
    return None


def _additive(state: CodegenState, update: Node, carry: Node) -> str | None:
    """Why ``update`` is not ``carry`` plus values that do not read it, with
    the carry read nowhere else in its graph, or None.

    The update adds or subtracts other values along a chain down to the
    carry, through copies and through runtime ``if``s whose branches each
    update it that way (or leave it).
    """
    chain: set[Node] = set()
    pending: list[Node] = [update]
    while pending:
        node = pending.pop()
        if node is carry or node in chain:
            continue
        chain.add(node)
        if node.target is _new_var or node.target is _phi:
            operands = [arg for arg in node.args[:2] if isinstance(arg, Node)]
            if node.target is _new_var:
                operands = operands[:1]
        elif node.target in _ADDITIVE_TARGETS:
            positions = [
                position
                for position, arg in enumerate(node.args[:2])
                if _reads(arg, carry)
            ]
            if (
                len(positions) != 1
                or (positions[0] == 1 and node.target is not torch.ops.aten.add.Tensor)
                or (positions[0] == 1 and node.kwargs.get("alpha", 1) != 1)
            ):
                return f"the loop updates it with {node.target}"
            operands = [cast("Node", node.args[positions[0]])]
        elif node.target is operator.getitem and _is_if(node.args[0]):
            branch_input = _branch_input(state, node, carry)
            if isinstance(branch_input, str):
                return branch_input
            if_node = cast("Node", node.args[0])
            chain.add(if_node)
            operands = [branch_input]
        else:
            return f"the loop updates it with {node.target}"
        if not operands or not all(_reads(arg, carry) for arg in operands):
            return "the loop does not update it by adding"
        pending.extend(operands)
    for node in (*chain, carry):
        if node is update or _is_if(node):
            # The output reads the update; a branch's reads are checked in
            # its own graph.
            continue
        if any(user not in chain for user in _element_users(node)):
            return "the loop reads it other than to add to it"
    return None


def _branch_input(state: CodegenState, item: Node, carry: Node) -> Node | str:
    """The ``if`` argument, a value of ``carry``, whose additive update the
    branch output ``item`` is, or why it is none."""
    from ..device_ir import IfGraphInfo

    if_node = cast("Node", item.args[0])
    index = cast("int", item.args[1])
    if_info = state.get_graph(if_node.args[1])
    assert isinstance(if_info, IfGraphInfo)
    entries = if_info.branches_outputs or []
    common = sum(1 for entry in entries if all(isinstance(e, int) for e in entry))
    if_only = sum(1 for entry in entries if isinstance(entry[1], str))
    else_only = len(entries) - common - if_only
    # The ``_if`` results: the if branch's common and if-only outputs, the
    # outer values the else branch replaces, then the else branch's common
    # outputs, the outer values the if branch replaces and its else-only
    # outputs (``WalkDeviceAST.visit_If``).
    if index < common + if_only:
        branch, entry = 0, entries[index]
    elif common + if_only + else_only <= index < 2 * common + if_only + else_only:
        branch, entry = 1, entries[index - (common + if_only + else_only)]
    elif index >= 2 * common + 2 * if_only + else_only:
        branch = 1
        entry = entries[
            common + if_only + index - (2 * common + 2 * if_only + else_only)
        ]
    else:
        return "an if passes it through unexpectedly"
    output_index = entry[branch]
    assert isinstance(output_index, int)
    graph = state.get_graph(if_node.args[1 + branch]).graph
    (output,) = graph.find_nodes(op="output")
    outputs = output.args[0]
    assert isinstance(outputs, (list, tuple))
    branch_update = outputs[output_index]
    args = if_node.args[3 + branch]
    assert isinstance(args, (list, tuple))
    inputs = [
        (placeholder, arg)
        for placeholder, arg in zip(
            graph.find_nodes(op="placeholder"), args, strict=True
        )
        if isinstance(arg, Node)
        and _reads(arg, carry)
        and _reads(branch_update, placeholder)
    ]
    if len(inputs) != 1 or not isinstance(branch_update, Node):
        return "an if replaces it"
    placeholder, arg = inputs[0]
    if not all(user.op == "output" for user in _element_users(branch_update)):
        return "an if reads it other than to add to it"
    reason = _additive(state, branch_update, placeholder)
    return arg if reason is None else f"in an if, {reason}"


def _is_if(node: object) -> bool:
    return isinstance(node, Node) and node.target is _if


def _is_zero(node: object) -> bool:
    """Whether ``node`` is a factory filled with zero."""
    if not isinstance(node, Node):
        return False
    if node.target in (full, torch.ops.aten.full.default):
        value = node.args[1]
        return isinstance(value, (int, float)) and value == 0
    return node.target in (
        torch.ops.aten.zeros.default,
        torch.ops.aten.new_zeros.default,
    )


def _skip_copies(node: object) -> object:
    while isinstance(node, Node) and node.target is _new_var:
        node = node.args[0]
    return node


def _element_users(node: Node) -> list[Node]:
    return [user for user in node.users if user.target not in _METADATA_TARGETS]


def _reads(arg: object, placeholder: Node) -> bool:
    """Whether ``arg`` depends on the elements of ``placeholder`` (its size,
    which a load of the same tile reads, is not its elements)."""
    if not isinstance(arg, Node):
        return False
    pending = [arg]
    seen: set[Node] = set()
    while pending:
        node = pending.pop()
        if node is placeholder:
            return True
        if node in seen or node.target in _METADATA_TARGETS:
            continue
        seen.add(node)
        pending.extend(node.all_input_nodes)
    return False


def _sums_dims(node: Node, dims: list[int]) -> bool:
    """Whether the sum ``node`` reduces every dim in ``dims`` of its input."""
    source = node.args[0]
    reduced = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if not isinstance(source, Node) or not isinstance(reduced, (list, tuple)):
        return False
    value = source.meta.get("val")
    assert isinstance(value, torch.Tensor)
    normalized = {cast("int", dim) % value.ndim for dim in reduced}
    return all(dim in normalized for dim in dims)
