from __future__ import annotations

import functools
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import sympy
from torch._inductor.bounds import ValueRangeAnalysis
from torch._inductor.ir import Loops
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
import torch.fx
from torch.fx.experimental import proxy_tensor
from torch.utils import _pytree as pytree
from torch.utils._sympy.value_ranges import ValueRanges

from ..language._tracing_ops import _if
from ..language._tracing_ops import _mask_select
from ..language._tracing_ops import _mask_to
from ..language._tracing_ops import _new_var
from ..language._tracing_ops import _phi
from ..language._tracing_ops import _while_loop
from ..language._tracing_ops import is_for_loop_target
from ..language.creation_ops import full
from .compile_environment import CompileEnvironment
from .compile_environment import _symint_expr

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence

    from .device_ir import GraphInfo
    from .inductor_lowering import InductorLowering

    ValueRangesAny = ValueRanges[Any]


# Relayout ops a load's out-of-bounds mask may be deferred *through*: the mask is
# re-materialized later, in the consumer's layout, by a downstream ``_mask_to``
# (see ``defer_pallas_load_masks``).
#
# Restricted to ops that permute tile axes WITHOUT regrouping the masked
# dimension's elements, so the masked dimension's set of valid/invalid lanes is
# preserved exactly (only its axis position changes).  ``permute`` is the only
# such op needed today -- ``transpose``/``.T`` lower to it.
#
# Deliberately NOT included:
#   * ``view``/``reshape``: even when the masked block id still appears exactly
#     once in the output shape, a reshape can regroup elements so the new
#     per-axis mask (``arange < extent``) selects different flat positions than
#     the eager load mask did -- e.g. a ``[B, 2]`` tile with 3 valid rows
#     reshaped to ``[2, B]`` (old invalid flat lanes 6,7 -> new mask zeroes 3,7,
#     dropping valid data and leaking invalid data).  "the dim survives once" is
#     necessary but NOT sufficient; admitting these needs a stride/lane-set
#     equivalence proof, not just a dim-count check.
#   * ``expand``/``stack``/``gather``: pass the masked *value* through but can
#     replicate or relocate padded lanes into valid ones.
#   * ``squeeze``/``unsqueeze``/``alias``: safe in principle (no regrouping) but
#     left out until they have their own deferral tests.
#
# IMPORTANT: every op here must be RANK-PRESERVING.  ``defer_pallas_load_masks``
# relies on that: its profitability gate (masked axis is a major dim at the load
# but a last-two dim at the consumer) doubles as the old "a relayout actually
# moved the axis" check *only* because a same-shape direct ``_mask_to`` cannot put
# an axis in both positions at once.  A rank-changing op (squeeze/unsqueeze/view/
# reshape) would break that, and an explicit relayout-crossed check would need to
# be reinstated.
_RELAYOUT_TARGETS = frozenset({torch.ops.aten.permute.default})

# Per-load map from deferred block id to the downstream ``_mask_to`` node names
# that discharge its mask. Pallas emit-pipeline lowering uses this proof to add
# a physical tensor-extent mask at the same cheap, post-relayout location when a
# shortened DMA can leave its VMEM tail stale.
PALLAS_DEFERRED_MASK_CONSUMERS_META = "pallas_deferred_mask_consumers"

# On a ``_for_loop`` node, for each carry with a dim on the loop's own blocks
# (by its index), those of the blocks along which its update reads the tile's
# lanes (``_LaneReads``), in all of it or in a branch of a runtime ``if``, as
# ``mask_loop_carries`` records them.  Only the CuTe lane-merged carry check
# reads it; the other backends ignore it.
CARRIES_READING_TILE_LANES_META = "carries_reading_tile_lanes"
# On a ``_for_loop`` node, the indices of the carries ``mask_loop_carries``
# leaves unmasked because a select of a whole update would drop what a
# branch adds to the padded lanes (``_LaneReads.select_drops_a_branch``).
# Only the CuTe lane-merged carry check reads it.
CARRIES_LEFT_UNMASKED_META = "carries_left_unmasked"


def mask_node_inputs(
    node: torch.fx.Node,
    other: float | bool = 0,
) -> None:
    """Inplace update the node's args and kwargs to apply masking."""
    apply = functools.partial(apply_masking, other=other, base_node=node)
    node.args = torch.fx.map_arg(node.args, apply)
    node.kwargs = torch.fx.map_arg(node.kwargs, apply)


def apply_masking(
    node: torch.fx.Node,
    *,
    base_node: torch.fx.Node,
    other: float | bool = 0,
) -> torch.fx.Node:
    """Analyze the node and apply masking."""
    for user in node.users:
        if user.op == "call_function" and user.target == _mask_to:
            # A mask made for a later consumer is not defined at this one.
            if user.args[1] == other and user < base_node:
                assert user.args[0] is node
                return user  # reuse existing mask_to node
    from .inductor_lowering import APIFuncLowering

    # If we reach here, we need to create a new mask_to node
    with node.graph.inserting_before(base_node):
        new_node = node.graph.call_function(_mask_to, (node, other), {})
    new_node.meta.update(base_node.meta)
    with proxy_tensor.disable_proxy_modes_tracing():
        new_node.meta["val"] = node.meta["val"].clone()
    new_node.meta["lowering"] = APIFuncLowering(_mask_to)
    return new_node


def remove_unnecessary_masking(graph: torch.fx.Graph) -> None:
    """Remove unnecessary _mask_to nodes from the graph."""
    from .inductor_lowering import ReductionLowering

    upstream_reduction_cache: dict[torch.fx.Node, bool] = {}
    downstream_reduction_cache: dict[torch.fx.Node, bool] = {}

    def depends_on_reduction_output(node: torch.fx.Node) -> bool:
        cached = upstream_reduction_cache.get(node)
        if cached is not None:
            return cached

        stack = [node]
        visited: set[torch.fx.Node] = set()
        result = False
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            lowering = current.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                result = True
                break
            stack.extend(current.all_input_nodes)

        upstream_reduction_cache[node] = result
        return result

    def feeds_reduction_input(node: torch.fx.Node) -> bool:
        cached = downstream_reduction_cache.get(node)
        if cached is not None:
            return cached

        result = False
        for user in node.users:
            lowering = user.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                result = True
                break
            if user.op == "call_function" and feeds_reduction_input(user):
                result = True
                break

        downstream_reduction_cache[node] = result
        return result

    # The device loops around ``graph``, found for the first mask that needs
    # them.
    loops: _EnclosingLoops | None = None
    for node in graph.find_nodes(op="call_function", target=_mask_to):
        input_node, masked_value0 = node.args
        assert isinstance(input_node, torch.fx.Node)
        masked_value1 = cached_masked_value(input_node)
        if masked_value0 == masked_value1:
            # If the value feeds a reduction and depends on a reduction output,
            # we must preserve the mask to zero-out padded lanes. For example, in
            # `test_layer_norm_nonpow2_reduction`, the 1536-wide reduction tile is padded to
            # the next power of two; the mask keeps those extra lanes at the
            # neutral value so the mean/variance sums stay correct. Rolled
            # reductions similarly insert a mask right before the final sum to
            # discard zero-padded iterations. We need to keep the mask in these
            # cases otherwise the padded lanes would contribute irrelevant data and
            # corrupt the reduction result.
            if feeds_reduction_input(input_node) and depends_on_reduction_output(
                input_node
            ):
                continue
            if loops is None:
                loops = _EnclosingLoops(graph)
            if _reduces_stale_carry_lanes(node, input_node, loops):
                continue
        else:
            for user in _dots_zeroed_by_partner(node, masked_value1):
                user.replace_input_with(node, input_node)
            if node.users:
                continue
        node.replace_all_uses_with(  # pyrefly: ignore [missing-attribute]
            input_node
        )
        graph.erase_node(node)


class _EnclosingLoops:
    """The device loops around a graph of the current device IR, for
    ``_reduces_stale_carry_lanes``.

    ``blocks`` holds their canonical block ids, the graph's own included
    (``blocks_around`` gives them for any graph).  ``scopes`` maps each graph
    that a ``_for_loop`` or an ``_if`` runs to the loop's canonical block ids
    (``None`` for a branch), the call and the slot of the call's arguments
    the graph's inputs take, ``for_loops`` maps each ``_for_loop`` call to
    its loop's block ids, and ``graphs`` each graph id to its graph.  The
    parents come from the control-flow calls (``control_flow_parent_ids``),
    so the walk does not depend on ``node_args``, which a graph without
    inputs lacks and a copied graph leaves pointing at the original's
    parent.  A root or helper graph, which no loop runs, skips the walk: no
    loop is around it."""

    def __init__(self, graph: torch.fx.Graph) -> None:
        from .device_ir import DeviceIR
        from .device_ir import ElseGraphInfo
        from .device_ir import ForLoopGraphInfo
        from .device_ir import HelperFunctionGraphInfo
        from .device_ir import IfGraphInfo
        from .device_ir import RootGraphInfo
        from .device_ir import control_flow_parent_entries
        from .device_ir import control_flow_parent_ids

        env = CompileEnvironment.current()
        self.env = env
        graphs = DeviceIR.current().graphs
        current = next((info for info in graphs if info.graph is graph), None)
        self.scopes: dict[
            torch.fx.Graph, tuple[frozenset[int] | None, torch.fx.Node, int]
        ] = {}
        self.for_loops: dict[torch.fx.Node, frozenset[int]] = {}
        self.graphs = {info.graph_id: info.graph for info in graphs}
        self.blocks: frozenset[int] = frozenset()
        self._by_id = {info.graph_id: info for info in graphs}
        # A rolled loop's bookkeeping copy shares its graph: the first
        # graph info of a graph is its owner, as ``current`` above is.
        self._by_graph = {info.graph: info for info in reversed(graphs)}
        self._parents: dict[int, int] = {}
        self._around: dict[torch.fx.Graph, frozenset[int]] = {}
        if current is None or isinstance(
            current, (RootGraphInfo, HelperFunctionGraphInfo)
        ):
            return
        for child_id, (call, slot) in control_flow_parent_entries(graphs).items():
            info = self._by_id.get(child_id)
            if isinstance(info, (ForLoopGraphInfo, IfGraphInfo, ElseGraphInfo)):
                own_blocks = self._own_blocks(info)
                self.scopes.setdefault(info.graph, (own_blocks, call, slot))
                if own_blocks is not None:
                    self.for_loops.setdefault(call, own_blocks)
        self._parents = control_flow_parent_ids(graphs)
        self.blocks = self.blocks_around(graph)

    def _own_blocks(self, info: object) -> frozenset[int] | None:
        """The canonical block ids of a device loop's body ``info``, or None
        for any other graph."""
        from .device_ir import ForLoopGraphInfo

        if not isinstance(info, ForLoopGraphInfo):
            return None
        return frozenset(
            self.env.canonical_block_id(block_id) for block_id in info.block_ids
        )

    def blocks_around(self, graph: torch.fx.Graph) -> frozenset[int]:
        """The canonical block ids of the device loops around ``graph``, its
        own included."""
        if (cached := self._around.get(graph)) is not None:
            return cached
        blocks: set[int] = set()
        info = self._by_graph.get(graph)
        while info is not None:
            blocks.update(self._own_blocks(info) or ())
            parent = self._parents.get(info.graph_id)
            info = None if parent is None else self._by_id.get(parent)
        self._around[graph] = frozenset(blocks)
        return self._around[graph]


def _reduces_stale_carry_lanes(
    mask: torch.fx.Node, input_node: torch.fx.Node, loops: _EnclosingLoops
) -> bool:
    """Whether ``mask``, the ``_mask_to`` of ``input_node`` in a device loop's
    body or in a graph nested in one, keeps a carry's stale lanes out of a
    reduction over the block of a loop around it.

    A carry with a dim on its own loop's block, like ``acc`` in ``acc = acc +
    x[tile_m, tile_n]``, takes the masked value of its initial value, but in
    the last iteration of a partial tile its padded lanes hold the earlier
    iterations' values, as do the loop's results after it.  A user of the
    mask reduces over the block of the dim it sums along or of the K axis an
    ``hl.dot`` contracts (``_reduced_blocks``); any other user counts as
    reducing over every loop block of the mask.  The walk goes back from
    ``input_node`` through every value but a reduction over a loop block in
    a graph inside every loop around the mask, whose own mask kept stale
    lanes along any of them out of it (one in an enclosing graph was decided
    over fewer loops, so its result may hold a nested loop's stale lanes
    along a block it does not reduce: ``c.sum(1)`` of a carry stale along
    the rows another loop over them sums), to stale lanes: a placeholder
    that may hold them (``_placeholder_sources``: a carry with a dim on its
    own loop's block, a while loop's state) or such a loop's result after it
    (``_stale_loop_result``).  A dim of either holds the lanes of each
    block whose size its size involves (``_spanned_blocks``): a merged dim
    too, like the ``tile_m * bn`` of ``acc.flatten()``.  The other
    results of nested loops and branches, whose bodies the walk also looks
    into, count as reading the inputs they are given."""

    def loop_blocks(node: torch.fx.Node) -> set[int]:
        return set(_dim_blocks(node.meta.get("val"), loops.blocks).values())

    masked = loop_blocks(mask)
    reduced: set[int] = set()
    for user in mask.users:
        blocks = _reduced_blocks(user, mask)
        reduced |= masked if blocks is None else masked & blocks
    if not reduced:
        return False
    pending = [input_node]
    seen: set[torch.fx.Node] = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if _reduction_block(node, loops.env) in loops.blocks and (
            loops.blocks <= loops.blocks_around(node.graph)
        ):
            continue
        blocks = _spanned_blocks(node, loops.blocks)
        if node.op == "placeholder":
            if blocks:
                sources = _placeholder_sources(node, blocks, loops)
                if sources is None:
                    return True
                pending.extend(sources)
            continue
        if _stale_loop_result(node, blocks, loops):
            return True
        if blocks:
            sources = _result_sources(node, loops)
            if sources is None:
                return True
            pending.extend(sources)
        pending.extend(node.all_input_nodes)
    return False


def _reduction_block(node: torch.fx.Node, env: CompileEnvironment) -> int | None:
    """The canonical block of the dim the reduction ``node`` sums along, or
    None when ``node`` is no reduction."""
    from .inductor_lowering import ReductionLowering

    lowering = node.meta.get("lowering")
    if not isinstance(lowering, ReductionLowering):
        return None
    return env.canonical_block_id(lowering.block_index)


def _reduced_blocks(user: torch.fx.Node, operand: torch.fx.Node) -> set[int] | None:
    """The canonical blocks that ``user`` reduces ``operand`` over: that of
    the dim a reduction sums along, or that of the K axis an ``hl.dot``
    contracts (the last of its first matrix, the second to last of its
    second).  None for any other user."""
    from ..language.matmul_ops import dot_operand_indices

    env = CompileEnvironment.current()
    if (block := _reduction_block(user, env)) is not None:
        return {block}
    indices = dot_operand_indices(user)
    value = operand.meta.get("val")
    if indices is None or not isinstance(value, torch.Tensor):
        return None
    lhs, rhs = (user.args[i] for i in indices)
    axes = [*([-1] if operand is lhs else []), *([-2] if operand is rhs else [])]
    return {
        env.canonical_block_id(block_id)
        for axis in axes
        if (block_id := env.get_block_id(value.shape[axis])) is not None
    }


def _spanned_blocks(node: torch.fx.Node, blocks: frozenset[int]) -> set[int]:
    """The canonical ids among ``blocks`` of the blocks whose lanes the dims
    of ``node``'s value hold, for ``_reduces_stale_carry_lanes``: those whose
    block size a dim's size involves.  That is a dim of the block's size,
    and a merged dim too: ``acc.flatten()`` of a ``[tile_m, bn]`` carry has
    one dim, ``tile_m * bn``, that holds the carry's lanes along ``bn``, and
    so its stale lanes in a branch or an inner loop given the flattened
    value, or as a flattened carry.  A size merely computed from a block
    size, like ``cdiv(n, bn)``, counts as well, which errs toward keeping
    masks."""
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return set()
    env = CompileEnvironment.current()
    return {
        canonical
        for size in value.shape
        if isinstance(size, torch.SymInt) and (expr := _symint_expr(size)) is not None
        for symbol in expr.free_symbols
        if (block_id := env.get_block_id(symbol)) is not None
        and (canonical := env.canonical_block_id(block_id)) in blocks
    }


def _stale_loop_result(
    result: torch.fx.Node, blocks: set[int], loops: _EnclosingLoops
) -> bool:
    """Whether ``result``, a value with dims on the loop blocks ``blocks``, is
    a ``_for_loop``'s result for a carry with a dim on that loop's own block,
    whose padded lanes hold the loop's iterations' values."""
    if result.target is not operator.getitem:
        return False
    own_blocks = loops.for_loops.get(cast("torch.fx.Node", result.args[0]))
    return bool(own_blocks and own_blocks & blocks) and any(
        user.target is _phi and user.args[1] is result for user in result.users
    )


def _result_sources(
    result: torch.fx.Node, loops: _EnclosingLoops
) -> list[torch.fx.Node] | None:
    """What ``result``, a value with dims on loop blocks, holds besides the
    inputs of the call it reads, for ``_reduces_stale_carry_lanes``: for a
    ``_for_loop``'s result, the body's update of that carry, and for an
    ``_if``'s, every output of its two branches (the walk does not tell
    which one gives the result).  None for the result of any other
    control-flow call, a while loop's, which may hold any iteration's lanes,
    and nothing for any other value."""
    from .device_ir import control_flow_child_graph_ids

    if result.target is not operator.getitem or not isinstance(
        call := result.args[0], torch.fx.Node
    ):
        return []
    if is_for_loop_target(call.target):
        graph_ids = [call.args[0]]
    elif call.target is _if:
        graph_ids = [call.args[1], call.args[2]]
    elif control_flow_child_graph_ids(call):
        return None
    else:
        return []
    sources: list[torch.fx.Node] = []
    for graph_id in graph_ids:
        graph = loops.graphs.get(cast("int", graph_id))
        if graph is None:
            return None
        (output,) = graph.find_nodes(op="output")
        if is_for_loop_target(call.target):
            # The body gives back each carry's update in the carry's slot.
            outputs = [
                cast("list[object]", output.args[0])[cast("int", result.args[1])]
            ]
        else:
            outputs = pytree.tree_leaves(output.args)
        sources.extend(each for each in outputs if isinstance(each, torch.fx.Node))
    return sources


def _placeholder_sources(
    placeholder: torch.fx.Node, blocks: set[int], loops: _EnclosingLoops
) -> list[torch.fx.Node] | None:
    """What ``placeholder``, an input of a branch or a loop body with dims on
    ``blocks``, holds for ``_reduces_stale_carry_lanes``: the value the call
    gives it and, for a ``_for_loop``'s carry, the body's update of it.  None
    when it may hold an earlier iteration's lanes: a carry with a dim on its
    own loop's block, or a while loop's state (an input of a graph that no
    ``_for_loop`` or ``_if`` runs)."""
    scope = loops.scopes.get(placeholder.graph)
    if scope is None:
        return None
    own_blocks, call, slot = scope
    index = placeholder.graph.find_nodes(op="placeholder").index(placeholder)
    given = cast("list[object]", call.args[slot])[index]
    sources = [given] if isinstance(given, torch.fx.Node) else []
    if own_blocks is None:
        return sources
    (output,) = placeholder.graph.find_nodes(op="output")
    updates = cast("list[object]", output.args[0])
    stale = bool(own_blocks & blocks)
    for result in call.users:
        if result.target is not operator.getitem:
            continue
        for phi in result.users:
            if phi.target is _phi and phi.args == (given, result):
                if stale:
                    return None
                update = updates[cast("int", result.args[1])]
                if isinstance(update, torch.fx.Node):
                    sources.append(update)
    return sources


def _dots_zeroed_by_partner(
    mask: torch.fx.Node, masked_value: float | bool | None
) -> list[torch.fx.Node]:
    """The ``hl.dot`` users taking ``mask`` as a matrix whose partner is
    already 0 past the K tile, so their padded lanes' products are 0 without it.

    That fails only for a lane holding inf or NaN.  A known one (a constant
    fill) keeps the mask; an unknown value (such as attention's probabilities,
    multiplied by loaded values) is taken to be finite, as every ``hl.dot``
    operand was before they were masked, which keeps such matmuls' graphs
    unchanged.
    """
    from ..language.matmul_ops import dot_operand_indices

    if mask.args[1] != 0 or (
        isinstance(masked_value, float) and not math.isfinite(masked_value)
    ):
        return []
    # A Pallas load whose mask was deferred to this one has stale lanes.
    source = mask.args[0]
    while isinstance(source, torch.fx.Node) and source.target in _RELAYOUT_TARGETS:
        source = source.args[0]
    if (
        not isinstance(source, torch.fx.Node)
        or "pallas_deferred_mask_block_ids" in source.meta
    ):
        return []
    dots = []
    for user in mask.users:
        indices = dot_operand_indices(user)
        if indices is None:
            continue
        lhs, rhs = (user.args[i] for i in indices)
        if mask is lhs:
            partner = rhs
        elif mask is rhs:
            partner = lhs
        else:
            continue
        if isinstance(partner, torch.fx.Node) and partner.target is _mask_to:
            partner = partner.args[0]
        if isinstance(partner, torch.fx.Node) and cached_masked_value(partner) == 0:
            dots.append(user)
    return dots


class LoopCarry(NamedTuple):
    """A value a device loop carries from one iteration to the next."""

    index: int
    """Its position among the loop body's outputs and the loop's results."""
    phi: torch.fx.Node
    """The ``_phi`` after the loop that merges its result with its init."""
    init: torch.fx.Node
    """Its value before the loop."""
    placeholder: torch.fx.Node
    """The body's input that holds it during an iteration."""
    update: object
    """The body's output that holds it for the next iteration."""
    result: torch.fx.Node
    """``getitem(loop, index)``, its value after the loop."""


def loop_carries(loop: torch.fx.Node, body: torch.fx.Graph) -> list[LoopCarry]:
    """The carries of the ``_for_loop`` or ``_while_loop`` node ``loop``,
    whose body is ``body``: the results a ``_phi`` merges with an input of
    the loop."""
    inits = loop.args[2 if loop.target is _while_loop else 3]
    assert isinstance(inits, (list, tuple))
    placeholders = body.find_nodes(op="placeholder")
    (output,) = body.find_nodes(op="output")
    outputs = output.args[0]
    assert isinstance(outputs, (list, tuple))
    carries = []
    for result in loop.users:
        if result.target is not operator.getitem:
            continue
        index = result.args[1]
        assert isinstance(index, int)
        for phi in result.users:
            init = phi.args[0]
            if (
                phi.target is _phi
                and phi.args[1] is result
                and isinstance(init, torch.fx.Node)
                and init in inits
            ):
                placeholder = placeholders[inits.index(init)]
                carries.append(
                    LoopCarry(index, phi, init, placeholder, outputs[index], result)
                )
    return carries


def block_dims(value: object, blocks: Collection[int]) -> list[int]:
    """The dims of ``value``, a fake tensor, whose size is the block size of
    one of ``blocks``, canonical block ids."""
    return list(_dim_blocks(value, blocks))


def _dim_blocks(value: object, blocks: Collection[int]) -> dict[int, int]:
    """The dims of ``value``, a fake tensor, whose size is the block size of
    one of ``blocks``, canonical block ids, each with that block."""
    if not isinstance(value, torch.Tensor):
        return {}
    env = CompileEnvironment.current()
    return {
        dim: canonical
        for dim, size in enumerate(value.shape)
        if (block_id := env.get_block_id(size)) is not None
        and (canonical := env.canonical_block_id(block_id)) in blocks
    }


# Reads of a value's metadata, not of its elements.
METADATA_TARGETS = (torch.ops.aten.sym_size.int, torch.ops.aten.sym_stride.int)

# A carry update ``op(carry, other)`` leaves the carry as it is wherever
# ``other`` holds the op's identity.  A division has one for its divisor
# only, 1 (``_update_identity``).
_UPDATE_IDENTITIES: dict[object, float] = {
    torch.ops.aten.add.Tensor: 0,
    torch.ops.aten.sub.Tensor: 0,
    torch.ops.aten.mul.Tensor: 1,
    torch.ops.aten.maximum.default: -math.inf,
    torch.ops.aten.minimum.default: math.inf,
}


def mask_loop_carries(graphs: Sequence[GraphInfo]) -> None:
    """Keep the fill values of a partial tile's loads out of loop carries.

    A carry with a dim on its loop's own block (``acc = hl.zeros([tile_m,
    bn])`` updated in ``for tile_n in hl.tile(n, block_size=bn)``) is updated
    lane by lane, so in the last iteration of a partial tile the padded lanes
    take what their loads filled in (``acc + torch.exp(x[tile_m, tile_n])``
    adds ``exp(0)``).  Nothing after the loop can take that back: the loop's
    mask does not exist there, and those lanes also hold the earlier
    iterations' valid values.  So in an update of such a carry that reads
    the tile's lanes (``_LaneReads``), each operand that reads them takes its
    op's identity in the padded lanes (``_LaneReads.mask_tile_operands``):
    ``acc + torch.exp(x[tile_m, tile_n])`` adds 0 there (``acc + x[tile_m,
    tile_n]`` already adds the load's 0), and the rest of the update still
    reaches them, so ``acc * s + torch.exp(v)`` with a factor ``s`` the same
    in every lane rescales them as it does the live lanes, and ``(acc +
    torch.exp(v)) / 2`` halves them.  Values added to the update one after
    another count as one sum, however it groups them: ``acc + x + 1.0``
    adds nothing to the padded lanes, as ``acc + (x + 1.0)`` does.  The
    operand's ``_mask_select(operand, identity, blocks)`` selects on the
    masks of the loop's own blocks and renders as the plain operand when
    none of their tiles is partial.  Where
    an operand that reads the tile's lanes meets an op without such an
    identity (a ``where``, a division by it), the whole update is selected
    instead, ``_mask_select(update, carry, blocks)``, which keeps the carried
    value in the padded lanes.  An update from the carry and
    values that are the same in every lane of the loop's blocks (``acc +
    1``, ``acc + x[tile_m, tile_n].sum(1, keepdim=True)``) still reaches
    every lane, as it always has.  Where the update merges the branches of a
    runtime ``if`` given the carry, each branch's update is judged on its own
    (``_LaneReads.updates``): only one of them runs, so in ``if flag: acc =
    acc + torch.exp(x[tile_m, tile_n]) else: acc = acc + 1`` the first masks
    its operand and the second still reaches every lane.  An update that
    also changes the carry outside such an ``if`` is judged as a whole, and
    when only a select of the whole update would mask it, that select would
    keep the carried value in the padded lanes on every path: if a branch
    that reads none of the tile's lanes, or one that passes the carry on,
    is merged into the update and nothing outside the ``if`` reads them
    either (``acc + 1.0`` in a branch, then ``acc = acc + 1.0`` after the
    ``if``), what that path adds would not reach them, so such an update is
    left as it is (``_LaneReads.select_drops_a_branch``): its padded lanes
    take what the program computes there, its loads' fill values included.

    A carry dim of a literal size (``hl.zeros([tile_m, 16])`` for ``hl.tile(n,
    block_size=16)``) is not on the loop's block, so its padded lanes still
    take the fill values (CuTe refuses such a carry); a dim of size ``bn =
    hl.register_block_size(n)`` is on it.  Backends that do not lower
    ``_mask_select`` keep their carries as they are
    (``Backend.lowers_mask_select``).  Each loop records the blocks along
    which each such carry's update reads the tile's lanes
    (``CARRIES_READING_TILE_LANES_META``), for checks that run on code
    generation's copies of the graphs.
    """
    from .device_ir import ForLoopGraphInfo

    env = CompileEnvironment.current()
    if not env.backend.lowers_mask_select:
        return
    lanes = _LaneReads(graphs)
    for loop in lanes.loops:
        if not is_for_loop_target(loop.target):
            continue
        body = graphs[cast("int", loop.args[0])]
        assert isinstance(body, ForLoopGraphInfo)
        blocks = {env.canonical_block_id(block_id) for block_id in body.block_ids}
        selected: set[tuple[torch.fx.Graph, int]] = set()
        reading: dict[int, frozenset[int]] = {}
        left: set[int] = set()
        for carry in loop_carries(loop, body.graph):
            if not block_dims(carry.placeholder.meta.get("val"), blocks):
                continue
            if isinstance(carry.update, torch.fx.Node):
                reading[carry.index] = frozenset(
                    block
                    for block in blocks
                    if lanes.reads_tile_lanes(carry.update, body.graph, (block,))
                )
            for update in lanes.updates(body.graph, carry.index, carry.placeholder):
                if (
                    (update.graph, update.index) not in selected
                    and lanes.reads_tile_lanes(update.value, body.graph, blocks)
                    and not lanes.mask_tile_operands(update, body.graph, blocks)
                ):
                    if lanes.select_drops_a_branch(update, body.graph, blocks):
                        left.add(carry.index)
                        continue
                    selected.add((update.graph, update.index))
                    _select_carry_update(update, blocks)
        loop.meta[CARRIES_READING_TILE_LANES_META] = reading
        loop.meta[CARRIES_LEFT_UNMASKED_META] = frozenset(left)


class _CarryUpdate(NamedTuple):
    """An update of a loop carry: output ``index`` of ``graph``, the loop's
    body or a branch of a runtime ``if`` in it, is ``value``, made from
    ``carry``, the carry's value where ``graph`` starts."""

    graph: torch.fx.Graph
    index: int
    value: torch.fx.Node
    carry: torch.fx.Node


def _select_carry_update(
    update: _CarryUpdate,
    blocks: Collection[int],
    other: torch.fx.Node | None = None,
) -> None:
    """Make ``update`` keep the carried value (or ``other``, a value of the
    carry's chain) in the masked lanes of ``blocks``' tiles."""
    from .inductor_lowering import APIFuncLowering

    value = update.value
    (output,) = update.graph.find_nodes(op="output")
    outputs = output.args[0]
    assert isinstance(outputs, (list, tuple)) and outputs[update.index] is value
    with update.graph.inserting_before(output):
        select = update.graph.call_function(
            _mask_select,
            (value, update.carry if other is None else other, tuple(sorted(blocks))),
            {},
        )
    select.meta["location"] = value.meta["location"]
    with proxy_tensor.disable_proxy_modes_tracing():
        select.meta["val"] = value.meta["val"].clone()
    select.meta["lowering"] = APIFuncLowering(_mask_select)
    output.args = (
        type(outputs)(
            select if i == update.index else out for i, out in enumerate(outputs)
        ),
    )


def skip_copies(value: object) -> object:
    """``value`` with the ``_new_var`` copies around it taken off."""
    while isinstance(value, torch.fx.Node) and value.target is _new_var:
        value = value.args[0]
    return value


class _LaneReads:
    """What a value of a loop body reads along the loop's blocks, through
    the branches and loops nested in the body."""

    def __init__(self, graphs: Sequence[GraphInfo]) -> None:
        self.graphs = graphs
        self.info_of = {info.graph: info for info in graphs}
        self.loops = [
            node
            for info in graphs
            for node in info.graph.nodes
            if node.op == "call_function"
            and (is_for_loop_target(node.target) or node.target is _while_loop)
        ]
        self.loop_of = {
            cast("int", loop.args[1 if loop.target is _while_loop else 0]): loop
            for loop in self.loops
        }

    def leaves(
        self,
        value: torch.fx.Node,
        body: torch.fx.Graph,
        blocks: Collection[int],
        inputs: Collection[torch.fx.Node] = (),
    ) -> list[torch.fx.Node]:
        """The values with a dim on ``blocks`` that ``value``, a value of the
        loop body ``body`` or of a graph nested in it, reads through other
        such values: loads and tile indices (random numbers derive from
        those), whose padded lanes hold fill values, and ``body``'s inputs,
        the carried values and loop invariants, whose lanes all hold data, as
        do ``inputs``.  A value without such a dim, such as a reduction over
        the tile, is the same in every lane, and reads of a value's size are
        not reads of its elements.  So is a view that splits or merges the
        block's dim (``x[tile_m, tile_n].reshape([tile_m, bn // 2, 2])``):
        what it reads keeps its fill values in the padded lanes, as does a
        carry dim of a literal size."""
        from ..language.memory_ops import load
        from ..language.tile_ops import tile_index

        leaves: list[torch.fx.Node] = []
        pending = [value]
        seen: set[torch.fx.Node] = set()
        while pending:
            node = pending.pop()
            val = node.meta.get("val")
            if (
                node in seen
                or node.target in METADATA_TARGETS
                # The results of a multi-output op are its getitems.
                or (not isinstance(val, (list, tuple)) and not block_dims(val, blocks))
            ):
                continue
            seen.add(node)
            if (
                node in inputs
                or node.target in (load, tile_index)
                or (node.op == "placeholder" and node.graph is body)
            ):
                leaves.append(node)
            else:
                pending.extend(self._reads(node))
        return leaves

    def _reads(self, node: torch.fx.Node) -> list[torch.fx.Node]:
        """The values ``node`` reads: its inputs, but for an input of a
        nested graph the value the graph is given (and for a carry of a
        nested loop, also its update), and for a result of a nested loop or
        branch the output that makes it."""
        from .device_ir import NodeArgsGraphInfo

        if node.op == "placeholder":
            info = self.info_of[node.graph]
            assert isinstance(info, NodeArgsGraphInfo)
            reads = [info.placeholder_to_outer_arg(node)]
            if (loop := self.loop_of.get(info.graph_id)) is not None:
                reads.extend(
                    carry.update
                    for carry in loop_carries(loop, info.graph)
                    if carry.placeholder is node
                    and isinstance(carry.update, torch.fx.Node)
                )
            return reads
        if node.target is operator.getitem and isinstance(
            source := node.args[0], torch.fx.Node
        ):
            index = cast("int", node.args[1])
            if is_for_loop_target(source.target):
                return self._output(cast("int", source.args[0]), index)
            if source.target is _while_loop:
                return self._output(cast("int", source.args[1]), index)
            if source.target is _if:
                origin = if_result_source(source, index)
                if isinstance(origin, tuple):
                    return self._output(*origin)
                return [origin] if origin is not None else []
        return node.all_input_nodes

    def _output(self, graph_id: int, index: int) -> list[torch.fx.Node]:
        value = _graph_outputs(self.graphs[graph_id].graph)[index]
        return [value] if isinstance(value, torch.fx.Node) else []

    def updates(
        self, graph: torch.fx.Graph, index: int, carry: torch.fx.Node
    ) -> list[_CarryUpdate]:
        """The updates of a carry that output ``index`` of ``graph`` makes
        from ``carry``, the carry's value where ``graph`` starts: that output,
        or, when it merges the branches of a runtime ``if`` given ``carry``,
        the updates each branch that writes it makes from its input of
        ``carry`` (the others pass ``carry`` on), since only one of the
        branches runs."""
        value = _graph_outputs(graph)[index]
        if not isinstance(value, torch.fx.Node):
            return []
        branches = self._branches(skip_copies(value), carry)
        if branches is None:
            return [_CarryUpdate(graph, index, value, carry)]
        return [update for branch in branches for update in self.updates(*branch)]

    def _branches(
        self, node: object, carry: torch.fx.Node
    ) -> list[tuple[torch.fx.Graph, int, torch.fx.Node]] | None:
        """For ``node``, a ``_phi`` merging results of one runtime ``if`` given
        ``carry`` with each other or with ``carry``, the branch outputs it
        merges, each with the branch's input of ``carry``; else None."""
        from .device_ir import NodeArgsGraphInfo

        if not isinstance(node, torch.fx.Node) or node.target is not _phi:
            return None
        branches = []
        if_nodes = set()
        for arg in node.args:
            if skip_copies(arg) is carry:
                continue
            if not (
                isinstance(arg, torch.fx.Node)
                and arg.target is operator.getitem
                and isinstance(if_node := arg.args[0], torch.fx.Node)
                and if_node.target is _if
            ):
                return None
            if_nodes.add(if_node)
            source = if_result_source(if_node, cast("int", arg.args[1]))
            if not isinstance(source, tuple):
                if skip_copies(source) is carry:
                    continue
                return None
            graph_id, index = source
            info = self.graphs[graph_id]
            assert isinstance(info, NodeArgsGraphInfo)
            inputs = [
                placeholder
                for placeholder in info.graph.find_nodes(op="placeholder")
                if skip_copies(info.placeholder_to_outer_arg(placeholder)) is carry
            ]
            if len(inputs) != 1:
                return None
            branches.append((info.graph, index, inputs[0]))
        return branches if len(if_nodes) == 1 else None

    def reads_tile_lanes(
        self, update: torch.fx.Node, body: torch.fx.Graph, blocks: Collection[int]
    ) -> bool:
        """Whether ``update`` reads a value whose padded lanes hold fill
        values (see ``leaves``)."""
        return any(
            leaf.op != "placeholder" for leaf in self.leaves(update, body, blocks)
        )

    def select_drops_a_branch(
        self, update: _CarryUpdate, body: torch.fx.Graph, blocks: Collection[int]
    ) -> bool:
        """Whether a select of the whole ``update`` (``_select_carry_update``)
        would keep the carried value in the padded lanes where a path of the
        update reads none of the tile's lanes: the update merges the results
        of a runtime ``if`` given the carry, with nothing between the merge
        and the update reading the tile's lanes, and one of the merged
        values (a branch's result, or the carry a branch passes on) reads
        none of them either.  That path changes the padded lanes as it does
        the live ones (``acc + 1.0``), which the select would undo."""
        carry = update.carry
        node: object = update.value
        while isinstance(node, torch.fx.Node) and node is not carry:
            if node.target is _new_var:
                node = node.args[0]
                continue
            links = [
                arg
                for arg in node.all_input_nodes
                if carry in self.leaves(arg, body, blocks, (carry,))
            ]
            if node.target is _phi:
                # A merge after a nested loop leaves no path of the update
                # without the tile's lanes: the loop's result reads them.
                return _merges_if_results(node) and any(
                    not self.reads_tile_lanes(arg, body, blocks) for arg in links
                )
            if len(links) != 1 or any(
                self.reads_tile_lanes(arg, body, blocks)
                for arg in node.all_input_nodes
                if arg is not links[0]
            ):
                return False
            node = links[0]
        return False

    def mask_tile_operands(
        self, update: _CarryUpdate, body: torch.fx.Graph, blocks: Collection[int]
    ) -> bool:
        """Keep the tile's fill values out of the padded lanes of ``update``,
        made from ``update.carry``, by masking its operands that read them,
        if its chain allows: whether it does, so that no select of the whole
        update is needed.

        The chain runs from the update down to the carry, through each op's
        one input that reads the carry.  An op whose other inputs read none
        of the tile's lanes (``reads_tile_lanes``) passes the chain on: they
        are the same in every lane of ``blocks`` (a constant, a rescale factor
        such as ``torch.exp(m_i - m_new)[:, None]``) or read only ``body``'s
        inputs, and apply to the padded lanes as they do to the live ones, so
        ``acc * 0.5``, ``(acc + x) / 2`` and an online logsumexp's ``acc *
        torch.exp(m_i - m_new)[:, None]`` decay or rescale them, as does a
        unary op or a cast of the chain.  An other operand that reads the
        tile's lanes takes the identity of its op (``_UPDATE_IDENTITIES``)
        there, ``_mask_select(operand, identity, blocks)``, unless its masked
        value already shows the identity: it is not broadcast along
        ``blocks`` and reads none of ``body``'s inputs, whose padded lanes
        hold the earlier iterations' values.  So ``acc + torch.exp(v -
        m_new[:, None])`` adds 0 there.

        The values added to the chain one after another (``_adds_to``) count
        as one sum, however the update groups them: once one of them reads
        the tile's lanes, none reaches the padded lanes, so ``acc + x + 1.0``
        adds 0 there, as ``acc + (x + 1.0)`` does.  Where masking operands
        cannot show that (a value of the sum is the same in every lane or
        reads only ``body``'s inputs, or one that reads the tile's lanes is
        broadcast along ``blocks`` or scaled by an ``alpha`` that is not
        finite), the sum takes the chain value it adds to there instead,
        ``_mask_select(sum, chain, blocks)``.

        False when an operand of another op that reads the tile's lanes has
        no identity in its dtype (a ``where``, a division by it, a
        ``torch.maximum`` that promotes an integer operand) or is broadcast
        along ``blocks``, or when the chain reaches the carry through two of
        an op's inputs or passes the merge of a runtime ``if`` that also
        changes the carry outside it.  A runtime ``if`` or a loop whose
        result the chain passes through, below the update (``acc = acc *
        0.5`` after an ``if`` that adds a tile's exp), counts as an op of
        the values it is given: the loads in its branches or body keep their
        fill values.

        Masking by lane position relies on a tile's live elements staying in
        its live lanes, as they do through elementwise ops, broadcasts,
        reductions over other dims, a sort (which puts the padding lanes
        last) and a gather whose indices stay within the tile.  An index past
        the tile's end (a reversal over the whole block) reads its padding as
        data, at a lane that depends on the block size."""
        carry = update.carry
        masks: list[tuple[torch.fx.Node, int, float | bool | torch.fx.Node]] = []
        sums: list[_Sum] = []
        current: _Sum | None = None
        # The op that reads ``node`` and its argument that does (None when
        # ``node`` is not one of its arguments), or None for the update.
        consumer: tuple[torch.fx.Node, int | None] | None = None
        node: object = update.value
        while node is not carry:
            if not isinstance(node, torch.fx.Node):
                return False
            if node.target is _new_var:
                consumer, node = (node, 0), node.args[0]
                continue
            links = [
                arg
                for arg in node.all_input_nodes
                if carry in self.leaves(arg, body, blocks, (carry,))
            ]
            if len(links) != 1:
                return False
            (link,) = links
            reading = [
                arg
                for arg in node.all_input_nodes
                if arg is not link and self.reads_tile_lanes(arg, body, blocks)
            ]
            link_position = next(
                (i for i, arg in enumerate(node.args) if arg is link), None
            )
            if _adds_to(node, link):
                assert link_position is not None
                if current is None:
                    current = _Sum(consumer, link)
                    sums.append(current)
                current.base = link
                other = node.args[1 - link_position]
                if not reading:
                    # A constant, a value the same in every lane or one of
                    # ``body``'s inputs, which no mask of it keeps out.
                    current.unmaskable = True
                elif reading != [other]:
                    return False
                else:
                    assert isinstance(other, torch.fx.Node)
                    current.reads_tile_lanes = True
                    identity = cast("float | bool", _update_identity(node, other))
                    alpha = node.kwargs.get("alpha", 1)
                    if link_position == 0 and not (
                        alpha == 1
                        or (isinstance(alpha, (int, float)) and math.isfinite(alpha))
                    ):
                        # ``alpha`` scales the operand, which keeps a 0 in the
                        # padded lanes only when finite.
                        current.unmaskable = True
                    elif self._fills_padded_lanes(other, node, identity, body, blocks):
                        if broadcasts_onto_masked_lanes(other, node, blocks):
                            # Its select, in its own shape, would leave the
                            # lanes the op broadcasts it onto.
                            current.unmaskable = True
                        else:
                            current.masks.append((node, 1 - link_position, identity))
                consumer, node = (node, link_position), link
                continue
            current = None
            if reading:
                operands = node.args[:2]
                if link not in operands or len(reading) != 1:
                    return False
                position = 1 - operands.index(link)
                (other,) = reading
                if position >= len(operands) or operands[position] is not other:
                    return False
                identity = _update_identity(node, other)
                if identity is None:
                    return False
                if self._fills_padded_lanes(other, node, identity, body, blocks):
                    if broadcasts_onto_masked_lanes(other, node, blocks):
                        # Its select, in its own shape, would leave the lanes
                        # the op broadcasts it onto.
                        return False
                    masks.append((node, position, identity))
            consumer, node = (node, link_position), link
        update_select: torch.fx.Node | None = None
        for total in sums:
            if not total.reads_tile_lanes:
                continue
            if not total.unmaskable:
                masks.extend(total.masks)
            elif total.consumer is None:
                update_select = total.base
            elif total.consumer[1] is None:
                return False
            else:
                masks.append((total.consumer[0], total.consumer[1], total.base))
        # A mask changes its op for every reader, so an op that another
        # output on ``blocks`` reads (another carry's update, as in ``v = (a
        # + (x + 1)) * b; a = v; b = v``) leaves this update to a select of
        # its own; masking it would change that output's padded lanes too.
        shared = _read_by_other_outputs(update, blocks)
        if any(op in shared for op, _, _ in masks):
            return False
        for op, position, identity in masks:
            _mask_operand(op, position, identity, tuple(sorted(blocks)))
        if update_select is not None:
            _select_carry_update(update, blocks, update_select)
        return True

    def _fills_padded_lanes(
        self,
        operand: torch.fx.Node,
        op: torch.fx.Node,
        identity: float | bool,
        body: torch.fx.Graph,
        blocks: Collection[int],
    ) -> bool:
        """Whether ``operand`` of ``op`` reads the tile's lanes and may hold
        something other than ``identity`` in the masked lanes of ``blocks``'
        tiles (``mask_tile_operands``)."""
        if operand.target is _mask_select or not self.reads_tile_lanes(
            operand, body, blocks
        ):
            return False
        return (
            cached_masked_value(operand) != identity
            or broadcasts_onto_masked_lanes(operand, op, blocks)
            or any(
                leaf.op == "placeholder" for leaf in self.leaves(operand, body, blocks)
            )
        )


def _merges_if_results(phi: torch.fx.Node) -> bool:
    """Whether the ``_phi`` node ``phi`` merges a result of a runtime ``if``
    (as ``_LaneReads._branches`` reads it), rather than a loop's result with
    the value before the loop."""
    return any(
        isinstance(arg, torch.fx.Node)
        and arg.target is operator.getitem
        and isinstance(arg.args[0], torch.fx.Node)
        and arg.args[0].target is _if
        for arg in phi.args
    )


def _read_by_other_outputs(
    update: _CarryUpdate, blocks: Collection[int]
) -> set[torch.fx.Node]:
    """The values that the outputs of ``update.graph`` with a dim on
    ``blocks``, other than ``update``'s, read in that graph."""
    pending = [
        out
        for index, out in enumerate(_graph_outputs(update.graph))
        if index != update.index
        and isinstance(out, torch.fx.Node)
        and block_dims(out.meta.get("val"), blocks)
    ]
    read: set[torch.fx.Node] = set()
    while pending:
        node = pending.pop()
        if node not in read:
            read.add(node)
            pending.extend(node.all_input_nodes)
    return read


def _update_identity(
    node: torch.fx.Node, operand: torch.fx.Node
) -> float | bool | None:
    """The identity of ``node``'s op (``_UPDATE_IDENTITIES``) for its
    ``operand``, in the operand's dtype, which its ``_mask_select`` writes, or
    None where that dtype has none: an integer's or a bool's bounds stand in
    for the infinities only when the op computes in the operand's dtype, as
    promoted (``torch.maximum`` of a float carry and an int32 load) they would
    bound nothing."""
    identity = _UPDATE_IDENTITIES.get(node.target)
    if node.target is torch.ops.aten.div.Tensor and node.args[1:2] == (operand,):
        # A division by 1 leaves its dividend as it is (``(acc * 0.5) /
        # torch.exp(x)``); no dividend leaves the divisor as it is.
        identity = 1
    value = operand.meta.get("val")
    if identity is None or not isinstance(value, torch.Tensor):
        return identity
    if value.dtype.is_floating_point or not math.isinf(identity):
        return bool(identity) if value.dtype is torch.bool else identity
    result = node.meta.get("val")
    if not isinstance(result, torch.Tensor) or result.dtype != value.dtype:
        return None
    if value.dtype is torch.bool:
        return identity > 0
    info = torch.iinfo(value.dtype)
    return info.max if identity > 0 else info.min


def _adds_to(node: torch.fx.Node, link: torch.fx.Node) -> bool:
    """Whether ``node`` adds a value to ``link``, or subtracts one from it,
    leaving ``link`` as it is: an ``add`` or a ``sub`` with ``link`` first, or
    an ``add`` with it second and no ``alpha`` (``torch.add(x, acc,
    alpha=2.0)`` scales ``acc``, and ``x - acc`` negates it)."""
    if node.target not in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor):
        return False
    first, second = node.args[:2]
    if (first is link) == (second is link):
        return False
    alpha = node.kwargs.get("alpha", 1)
    return first is link or (
        node.target is torch.ops.aten.add.Tensor
        and isinstance(alpha, (int, float))
        and alpha == 1
    )


class _Sum:
    """Values added to a carry update's chain one after another, as in ``acc
    + x + 1.0`` (``_LaneReads.mask_tile_operands``), down to ``base``, the
    chain value the first one is added to."""

    def __init__(
        self, consumer: tuple[torch.fx.Node, int | None] | None, base: torch.fx.Node
    ) -> None:
        # The op that reads the sum and its argument that does (None when the
        # sum is not one of its arguments), or None when the sum is the update.
        self.consumer = consumer
        self.base = base
        # The masks of its values that read the tile's lanes.
        self.masks: list[tuple[torch.fx.Node, int, float | bool]] = []
        # Whether one of its values reads the tile's lanes, and whether one
        # would reach the padded lanes in spite of those masks.
        self.reads_tile_lanes = False
        self.unmaskable = False


def _mask_operand(
    node: torch.fx.Node,
    position: int,
    identity: float | bool | torch.fx.Node,
    blocks: tuple[int, ...],
) -> None:
    """Make operand ``position`` of ``node`` take ``identity``, a value of
    its op's identity or of the carry's chain, in the masked lanes of
    ``blocks``' tiles (``_LaneReads.mask_tile_operands``)."""
    from .inductor_lowering import APIFuncLowering

    operand = node.args[position]
    assert isinstance(operand, torch.fx.Node)
    with node.graph.inserting_before(node):
        select = node.graph.call_function(_mask_select, (operand, identity, blocks), {})
    select.meta["location"] = node.meta["location"]
    with proxy_tensor.disable_proxy_modes_tracing():
        select.meta["val"] = operand.meta["val"].clone()
    select.meta["lowering"] = APIFuncLowering(_mask_select)
    node.update_arg(position, select)


def _graph_outputs(graph: torch.fx.Graph) -> list[object]:
    (output,) = graph.find_nodes(op="output")
    outputs, _ = pytree.tree_flatten(output.args)
    return outputs


def cached_masked_value(
    node: torch.fx.Node,
) -> float | bool | None:
    """Determine the current masked value for the node."""
    if "masked_value" in node.meta:
        return node.meta["masked_value"]

    if node.op == "placeholder":
        from .device_ir import DeviceIR
        from .device_ir import ForLoopGraphInfo
        from .device_ir import NodeArgsGraphInfo

        """
        We are inside a for loop or an if statement, which is represented as a subgraph.
        Let the analysis flow into the parent graph to find the masked value.
        """
        device_ir = DeviceIR.current()
        for graph_info in device_ir.graphs:
            if node.graph is graph_info.graph and isinstance(
                graph_info, NodeArgsGraphInfo
            ):
                outer_node = graph_info.placeholder_to_outer_arg(node)
                node.meta["masked_value"] = result = cached_masked_value(outer_node)
                if result is not None and isinstance(graph_info, ForLoopGraphInfo):
                    # check if the loop carry dependency is different
                    for user in outer_node.users:
                        if user.op == "call_function" and user.target == _phi:
                            loop_carry_result = cached_masked_value(user)
                            if loop_carry_result != result:
                                node.meta["masked_value"] = result = None
                                recompute_masked_values(node.graph)
                return result
        return None
    if node.op != "call_function":
        return None
    node.meta["masked_value"] = result = node.meta["lowering"].get_masked_value(node)
    return result


def recompute_masked_values(graph: torch.fx.Graph) -> None:
    """
    Recompute the masked values for all nodes in the graph.
    This is necessary when the loop carry dependencies change the mask value of an input node.
    """
    for node in graph.nodes:
        if node.op != "placeholder" and node.meta.get("masked_value") is not None:
            del node.meta["masked_value"]
            node.meta["masked_value"] = cached_masked_value(node)


def defer_pallas_load_masks(graph: torch.fx.Graph) -> None:
    """Defer a Pallas load's eager out-of-bounds mask to a downstream ``_mask_to``.

    Pallas load codegen materializes a tile's out-of-bounds mask multiplicatively
    in the load's own layout (``ref[idx] * mask``).  When the loaded value is only
    *relayouted* (an axis permutation; see ``_RELAYOUT_TARGETS``) and then consumed
    by a dot or reduction, that consumer already inserts a ``_mask_to(x, 0)`` which
    can re-materialize the same mask in the consumer's layout.  The mask is dynamic
    (``arange < extent``), so it cannot be elided even when logically all-true;
    applying it in the pre-relayout layout therefore keeps a live op on the path
    into the relayout.

    For each load whose masked tile dim is provably re-masked downstream, we:

    * record the deferred block ids on the load so Pallas load codegen skips the
      eager mask for those dims, and
    * mark the load's masked value unknown so ``remove_unnecessary_masking`` keeps
      the downstream ``_mask_to`` (it is no longer redundant once the load is not
      pre-masked).

    Correctness rests on a single dataflow fact: *every* use of the load reaches a
    ``_mask_to(_, 0)`` crossing only ``_RELAYOUT_TARGETS`` ops, with the masked
    dim still present as a standalone tile dim at each step.  Because those ops
    only permute axes (they do not regroup the masked dim's elements), the later
    per-axis mask covers exactly the lanes the eager mask would have.  The
    standalone-dim check is necessary but not sufficient on its own -- the
    correctness guarantee comes from restricting the crossed ops to pure axis
    permutations (so e.g. ``reshape`` is excluded; see ``_RELAYOUT_TARGETS``).  A
    use that does not re-mask (store, elementwise, reduction without a mask) keeps
    the eager load mask.

    Profitability is a *positional* gate on top of that correctness proof.  A mask
    on an axis inside the last-two (sublane/lane) dims is a vectorized per-register
    op, while a mask on a major (outer) axis is applied per outer row and is much
    more work.  So defer only when the masked axis is a major dim at the load and
    the relayout carries it into the last-two dims at the consumer ``_mask_to``;
    deferring in the reverse direction would move the mask onto the more expensive
    axis, so it is not done.  This gate also subsumes the "a relayout actually
    moved the axis" requirement (see the loop body).

    Pallas-only: Triton masks loads as real data (``tl.load(..., other=0)``), so
    relayout never moves unmasked lanes and there is nothing to defer.
    """
    from ..language.memory_ops import load as load_op
    from .aten_lowering import passthrough_masked_value

    env = CompileEnvironment.current()

    def dim_index(node: torch.fx.Node, block_id: int) -> int | None:
        """Index of ``block_id`` in ``node``'s value, or None unless it appears as
        exactly one standalone dimension (this doubles as the survives-uniquely
        check)."""
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return None
        hits = [
            i
            for i, size in enumerate(val.size())
            if env.resolve_block_id(size) == block_id
        ]
        return hits[0] if len(hits) == 1 else None

    def is_major_dim(node: torch.fx.Node, block_id: int) -> bool:
        # An outer dim, outside the last-two (sublane, lane) vreg tile, where a
        # mask is applied per outer row rather than as a per-register op.
        idx = dim_index(node, block_id)
        return idx is not None and idx < node.meta["val"].ndim - 2

    def is_last_two_dim(node: torch.fx.Node, block_id: int) -> bool:
        # Inside the last-two (sublane/lane) vreg tile, where a mask is a
        # vectorized per-register op.
        idx = dim_index(node, block_id)
        return idx is not None and idx >= node.meta["val"].ndim - 2

    def is_relayout(node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target not in _RELAYOUT_TARGETS:
            return False
        lowering = node.meta.get("lowering")
        return getattr(lowering, "masked_value_fn", None) is passthrough_masked_value

    def is_remask(node: torch.fx.Node, src: torch.fx.Node, block_id: int) -> bool:
        # A zero-fill ``_mask_to`` on ``src`` that re-masks ``block_id`` with that
        # axis in the last-two dims (the profitable place to apply the mask).
        # ``bool(...)``: ``node.args[1] == 0`` is typed as ``Argument`` (the fill is
        # always a scalar here, but the static type is a union), so coerce to bool.
        return bool(
            node.op == "call_function"
            and node.target is _mask_to
            and node.args[0] is src
            and node.args[1] == 0
            and is_last_two_dim(node, block_id)
        )

    def all_uses_remask(
        node: torch.fx.Node, block_id: int, memo: dict[torch.fx.Node, bool]
    ) -> bool:
        cached = memo.get(node)
        if cached is not None:
            return cached
        memo[node] = False  # conservative guard against revisiting mid-walk
        users = list(node.users)
        result = bool(users)
        for user in users:
            if is_remask(user, node, block_id):
                continue
            if (
                is_relayout(user)
                and dim_index(user, block_id) is not None
                and all_uses_remask(user, block_id, memo)
            ):
                continue
            result = False
            break
        memo[node] = result
        return result

    def collect_remask_consumers(
        node: torch.fx.Node,
        block_id: int,
        seen: set[torch.fx.Node],
    ) -> set[torch.fx.Node]:
        """Collect terminal zero-fill masks after ``all_uses_remask`` succeeds."""
        if node in seen:
            return set()
        seen.add(node)
        result: set[torch.fx.Node] = set()
        for user in node.users:
            if is_remask(user, node, block_id):
                result.add(user)
            elif is_relayout(user):
                result.update(collect_remask_consumers(user, block_id, seen))
        return result

    changed = False
    for node in graph.find_nodes(op="call_function", target=load_op):
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        candidates = {
            block_id
            for size in val.size()
            if (block_id := env.resolve_block_id(size)) is not None
        }
        deferred: set[int] = set()
        deferred_consumers: dict[int, tuple[str, ...]] = {}
        for block_id in candidates:
            # Profitability gate: defer only when the masked axis is a major/outer
            # dim at the load and a relayout carries it into the last-two
            # (vreg-tile) dims at the consumer ``_mask_to``.  A mask on a last-two
            # axis is a per-register op; a mask on a major axis is applied per
            # outer row, so this is the only direction that moves the mask onto a
            # cheaper axis (the reverse would move it onto a more expensive one).
            #
            # This also subsumes the "must cross >=1 relayout" check: with a
            # rank-preserving relayout set, a direct ``_mask_to`` shares the load's
            # shape, so ``block_id`` cannot be both major at the load and last-two
            # at the consumer (see the note on ``_RELAYOUT_TARGETS``).
            if not is_major_dim(node, block_id):
                continue
            if not node.users:
                continue
            if all_uses_remask(node, block_id, {}):
                deferred.add(block_id)
                deferred_consumers[block_id] = tuple(
                    sorted(
                        consumer.name
                        for consumer in collect_remask_consumers(node, block_id, set())
                    )
                )
        if deferred:
            node.meta["pallas_deferred_mask_block_ids"] = frozenset(deferred)
            node.meta[PALLAS_DEFERRED_MASK_CONSUMERS_META] = deferred_consumers
            node.meta["masked_value"] = None
            changed = True

    if changed:
        # Drop stale masked-value caches that assumed the load was pre-masked, so
        # the surviving ``_mask_to`` nodes are not wrongly judged redundant.
        recompute_masked_values(graph)


def getitem_masked_value(
    getitem_node: torch.fx.Node,
) -> float | bool | None:
    """
    Retrieve the masked value for a node that is a getitem operation.
    This handles loop outputs, since the `_for` node has multiple outputs.
    """
    from .device_ir import DeviceIR

    assert not getitem_node.kwargs, "getitem kwargs not supported"
    node, index = getitem_node.args
    assert isinstance(node, torch.fx.Node)
    assert isinstance(index, int)
    if is_for_loop_target(node.target):
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        source: tuple[int, int] | torch.fx.Node | None = (graph_id, index)
    elif node.target is _if:
        source = if_result_source(node, index)
    else:
        return None
    if not isinstance(source, tuple):
        # The outer value one branch passes through unchanged.
        return cached_masked_value(source) if source is not None else None
    graph_id, output_index = source
    (output_node,) = DeviceIR.current().graphs[graph_id].graph.find_nodes(op="output")
    outputs, _ = pytree.tree_flatten(output_node.args)
    output = outputs[output_index]
    if isinstance(output, torch.fx.Node):
        # TODO(jansel): need to pass cached_masked_value through to the inputs
        return cached_masked_value(output)
    return None


def if_result_source(
    if_node: torch.fx.Node,
    index: int,
    graphs: Sequence[GraphInfo] | None = None,
) -> tuple[int, int] | torch.fx.Node | None:
    """``IfGraphInfo.result_source`` of result ``index`` of the ``_if`` node
    ``if_node``, whose branches ``graphs`` holds by graph id (by default, the
    current ``DeviceIR``'s graphs)."""
    from .device_ir import DeviceIR
    from .device_ir import IfGraphInfo

    if_graph_id = if_node.args[1]
    assert isinstance(if_graph_id, int)
    if_info = (DeviceIR.current().graphs if graphs is None else graphs)[if_graph_id]
    assert isinstance(if_info, IfGraphInfo)
    return if_info.result_source(if_node, index)


class MaskedValueAnalysisInductor(ValueRangeAnalysis):
    def __init__(
        self, input_name_lookup: dict[str, ValueRangesAny], *, strict_float: bool
    ) -> None:
        super().__init__()
        self.input_name_lookup = input_name_lookup
        self.strict_float = strict_float

    def mul(
        self, a: ValueRangesAny | float, b: ValueRangesAny | float
    ) -> ValueRangesAny:
        a_range: ValueRangesAny = ValueRanges.wrap(a)
        b_range: ValueRangesAny = ValueRanges.wrap(b)
        # Inductor takes 0 * anything to be 0, but a float 0 times inf or NaN
        # is NaN (fast math assumes those away), so a factor that may not be
        # finite leaves the product unknown.
        if self.strict_float and not all(
            bound.is_finite
            for bound in (a_range.lower, a_range.upper, b_range.lower, b_range.upper)
        ):
            return ValueRanges.unknown()
        return super().mul(a_range, b_range)

    def load(self, name: str, index: sympy.Expr) -> ValueRangesAny:
        return self.input_name_lookup[name]

    @classmethod
    def index_expr(cls, index: object, dtype: torch.dtype) -> ValueRangesAny:
        return ValueRanges.unknown()


def inductor_masked_value(
    lowering: InductorLowering,
    node: torch.fx.Node,
) -> float | bool | None:
    """
    This analysis is used to determine the masked value inductor IR nodes.
    If the masked value of X is 0, then `X + 1` will be masked to 1.
    """

    pointwise = not isinstance(lowering.buffer.data, Reduction)

    def visit(n: torch.fx.Node) -> None:
        val = cached_masked_value(n)
        if val is None or (pointwise and broadcasts_onto_masked_lanes(n, node)):
            input_ranges.append(ValueRanges.unknown())
        else:
            input_ranges.append(ValueRanges(val, val))

    input_ranges: list[ValueRangesAny] = []
    for source in lowering.input_nodes(node):
        visit(source)
    val = node.meta.get("val")
    strict_float = (
        isinstance(val, torch.Tensor)
        and val.dtype.is_floating_point
        and not CompileEnvironment.current().settings.fast_math
    )
    with V.set_ops_handler(
        MaskedValueAnalysisInductor(
            dict(zip(lowering.input_names, input_ranges, strict=True)),
            strict_float=strict_float,
        )
    ):
        result = call_inner_fn(lowering.buffer.data)
        if result.is_singleton():
            val = result.lower
            if isinstance(val, (int, sympy.Integer)):
                return int(val)
            if isinstance(val, (float, sympy.Float)):
                return float(val)
        return None


def broadcasts_onto_masked_lanes(
    input_node: torch.fx.Node,
    node: torch.fx.Node,
    blocks: Collection[int] | None = None,
) -> bool:
    """Whether ``node`` broadcasts ``input_node`` along a dim that is not 1
    (one on the canonical ``blocks``, if given), and so may be masked.  The
    lanes past that dim's extent then hold the input's real values rather
    than its masked value, unless it is a constant fill."""
    if input_node.target is full:
        return False
    input_val = input_node.meta.get("val")
    output_val = node.meta.get("val")
    if not isinstance(input_val, torch.Tensor) or not isinstance(
        output_val, torch.Tensor
    ):
        return False
    env = CompileEnvironment.current()
    pad = output_val.ndim - input_val.ndim
    dims = range(output_val.ndim) if blocks is None else block_dims(output_val, blocks)
    return any(
        not env.known_equal(output_val.size(dim), 1)
        and (dim < pad or env.known_equal(input_val.size(dim - pad), 1))
        for dim in dims
    )


def call_inner_fn(loops: Loops) -> ValueRangesAny:
    indices = [sympy.Symbol(f"i{n}") for n in range(len(loops.ranges))]
    if isinstance(loops, Reduction):
        reduction_indices = [
            sympy.Symbol(f"r{n}") for n in range(len(loops.reduction_ranges))
        ]
        result = loops.inner_fn(indices, reduction_indices)
    else:
        result = loops.inner_fn(indices)
    if isinstance(result, OpsValue):
        result = result.value
    assert isinstance(result, ValueRanges)
    return result
