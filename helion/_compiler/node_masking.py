from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any
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
from torch.utils._sympy.value_ranges import ValueRanges

from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _if
from ..language._tracing_ops import _mask_to
from ..language._tracing_ops import _phi

if TYPE_CHECKING:
    from .inductor_lowering import InductorLowering

    ValueRangesAny = ValueRanges[Any]


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
            node.replace_all_uses_with(input_node)  # pyright: ignore[reportArgumentType]
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
    if node.target is _for_loop:
        graph_id = node.args[0]
    elif node.target is _if:
        graph_id = node.args[1]
    else:
        return None
    assert isinstance(graph_id, int)
    graph = DeviceIR.current().graphs[graph_id].graph
    (output_node,) = graph.find_nodes(op="output")
    (outputs,) = output_node.args
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
