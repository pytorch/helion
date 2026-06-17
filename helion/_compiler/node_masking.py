from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from typing_extensions import Never

import sympy
from torch._inductor.bounds import ValueRangeAnalysis
from torch._inductor.ir import Loops
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
import torch.fx
from torch.fx import map_arg
from torch.fx.experimental import proxy_tensor
from torch.utils import _pytree as pytree
from torch.utils._sympy.value_ranges import ValueRanges

from ..language._tracing_ops import _if
from ..language._tracing_ops import _mask_to
from ..language._tracing_ops import _phi
from ..language._tracing_ops import is_for_loop_target

if TYPE_CHECKING:
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
            if user.args[1] == other:
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
            node.replace_all_uses_with(  # pyrefly: ignore [missing-attribute]
                input_node
            )
            graph.erase_node(node)


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
    from .compile_environment import CompileEnvironment

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
        if deferred:
            node.meta["pallas_deferred_mask_block_ids"] = frozenset(deferred)
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
        graph_ids = [node.args[0]]
    elif node.target is _if:
        graph_ids = [node.args[1], node.args[2]]
    else:
        return None
    assert isinstance(graph_ids, list)
    assert all(isinstance(graph_id, int) for graph_id in graph_ids)
    graphs = [
        DeviceIR.current().graphs[cast("int", graph_id)].graph for graph_id in graph_ids
    ]
    output_nodes = [graph.find_nodes(op="output") for graph in graphs]
    outputs_all = [output[0].args for output in output_nodes]
    outputs, _ = pytree.tree_flatten(outputs_all)
    assert isinstance(outputs, (list, tuple))
    output = outputs[index]
    if isinstance(output, torch.fx.Node):
        # TODO(jansel): need to pass cached_masked_value through to the inputs
        return cached_masked_value(output)
    return None


class MaskedValueAnalysisInductor(ValueRangeAnalysis):
    def __init__(self, input_name_lookup: dict[str, ValueRangesAny]) -> None:
        super().__init__()
        self.input_name_lookup = input_name_lookup

    def load(self, name: str, index: sympy.Expr) -> ValueRangesAny:
        return self.input_name_lookup[name]

    @classmethod
    def index_expr(cls, index: Never, dtype: torch.dtype) -> ValueRangesAny:
        return ValueRanges.unknown()


def inductor_masked_value(
    lowering: InductorLowering,
    node: torch.fx.Node,
) -> float | bool | None:
    """
    This analysis is used to determine the masked value inductor IR nodes.
    If the masked value of X is 0, then `X + 1` will be masked to 1.
    """

    def visit(n: torch.fx.Node) -> torch.fx.Node:
        val = cached_masked_value(n)
        if val is None:
            input_ranges.append(ValueRanges.unknown())
        else:
            input_ranges.append(ValueRanges(val, val))
        return n

    input_ranges: list[ValueRangesAny] = []
    map_arg((node.args, node.kwargs), visit)
    with V.set_ops_handler(
        MaskedValueAnalysisInductor(
            dict(zip(lowering.input_names, input_ranges, strict=True)),
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
