"""Graph-owned contracts for regions containing computed contractions.

Discovery records the original operations, tensor domains and loop ports. It
does not choose a schedule or prove that arbitrary effects can be reordered.
Consumers must separately admit the surrounding operations, memory aliases and
physical layouts. In particular, tensor domains are logical extents, not padded
MMA tiles; predicates remain on the original load/store and pointwise nodes.

A region belongs to one graph revision. Recollect it after graph rewrites rather
than transplanting its node identities into another graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import cast

import sympy
import torch
from torch.fx import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..device_ir import ForLoopGraphInfo
from ..matmul_utils import _compute_out_dtype

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from ..device_ir import LoopInterface


LogicalExtent: TypeAlias = int | sympy.Expr
LogicalDomain: TypeAlias = tuple[LogicalExtent, ...]


@dataclass(frozen=True)
class ContractionSpec:
    """One dot, including its explicit numerical and logical-domain boundaries."""

    node: Node
    lhs: Node
    rhs: Node
    accumulator: Node | None
    operand_dtypes: tuple[torch.dtype, torch.dtype]
    result_dtype: torch.dtype
    accumulator_dtype: torch.dtype | None
    requested_out_dtype: torch.dtype | None
    lhs_domain: LogicalDomain
    rhs_domain: LogicalDomain
    result_domain: LogicalDomain

    @property
    def shape(self) -> tuple[LogicalExtent, LogicalExtent, LogicalExtent]:
        """Logical M, N, K; resolving physical tiles is a separate operation."""
        return self.lhs_domain[0], self.rhs_domain[1], self.lhs_domain[1]


@dataclass(frozen=True)
class ContractionCarry:
    """A traced loop port bound to original body nodes, never inferred by name."""

    input_index: int
    output_index: int
    input: Node
    output: Node


@dataclass(frozen=True)
class ContractionRegion:
    graph_id: int
    graph: torch.fx.Graph
    nodes: tuple[Node, ...]
    contractions: tuple[ContractionSpec, ...]
    live_ins: tuple[Node, ...]
    live_outs: tuple[Node, ...]
    loads: tuple[Node, ...]
    stores: tuple[Node, ...]
    scans: tuple[Node, ...]
    reductions: tuple[Node, ...]
    loop_interface: LoopInterface | None
    carries: tuple[ContractionCarry, ...]


def _domain(value: torch.Tensor) -> LogicalDomain:
    # Comparing SymInts directly can install guards or specialize a hint.
    return tuple(
        cast("sympy.Expr", size.node.expr) if isinstance(size, torch.SymInt) else size
        for size in value.shape
    )


def _same_extent(left: LogicalExtent, right: LogicalExtent) -> bool:
    if left == right:
        return True
    if not CompileEnvironment.has_current():
        return False
    env = CompileEnvironment.current()
    replacements: dict[sympy.Symbol, sympy.Expr] = {}
    difference = sympy.sympify(left) - sympy.sympify(right)
    for symbol in difference.free_symbols:
        block_id = env.get_block_id(symbol)
        if block_id is None:
            continue
        block = env.block_sizes[env.canonical_block_id(block_id)]
        source = block.block_size_source
        if isinstance(source, FixedBlockSizeSource) and type(source.value) is int:
            replacements[symbol] = sympy.Integer(source.value)
        else:
            replacements[symbol] = cast("sympy.Expr", block.var.node.expr)
    difference = difference.xreplace(replacements)
    if difference == 0:
        return True
    return env.shape_env._maybe_evaluate_static(sympy.Eq(difference, 0)) is sympy.true


def _same_domain(left: LogicalDomain, right: LogicalDomain) -> bool:
    return len(left) == len(right) and all(
        starmap(_same_extent, zip(left, right, strict=True))
    )


def _contraction(node: Node) -> ContractionSpec | None:
    # Helion's tracing normalization materializes all four dot arguments.
    if len(node.args) != 4 or node.kwargs:
        return None
    lhs, rhs, accumulator, out_dtype = node.args
    if (
        not isinstance(lhs, Node)
        or not isinstance(rhs, Node)
        or accumulator is not None
        and not isinstance(accumulator, Node)
        or out_dtype is not None
        and not isinstance(out_dtype, torch.dtype)
    ):
        return None
    left, right, result = (item.meta.get("val") for item in (lhs, rhs, node))
    if not all(isinstance(value, torch.Tensor) for value in (left, right, result)):
        return None
    assert isinstance(left, torch.Tensor)
    assert isinstance(right, torch.Tensor)
    assert isinstance(result, torch.Tensor)
    if left.ndim != 2 or right.ndim != 2 or result.ndim != 2:
        return None
    left_domain, right_domain, result_domain = map(_domain, (left, right, result))
    if (
        not _same_extent(left_domain[1], right_domain[0])
        or not _same_domain(result_domain, (left_domain[0], right_domain[1]))
        or left.device != right.device
        or left.device != result.device
    ):
        return None
    accumulator_dtype = None
    if accumulator is not None:
        acc = accumulator.meta.get("val")
        if (
            not isinstance(acc, torch.Tensor)
            or not _same_domain(_domain(acc), result_domain)
            or acc.device != result.device
            or acc.dtype not in (torch.float16, torch.float32, torch.int32)
        ):
            return None
        accumulator_dtype = acc.dtype
    # This is the traced result contract, not permission to fuse an addition
    # or to remove casts around an explicit accumulator.
    expected_dtype = (
        accumulator_dtype or out_dtype or _compute_out_dtype(left.dtype, right.dtype)
    )
    if result.dtype != expected_dtype:
        return None
    return ContractionSpec(
        node,
        lhs,
        rhs,
        accumulator,
        (left.dtype, right.dtype),
        result.dtype,
        accumulator_dtype,
        out_dtype,
        left_domain,
        right_domain,
        result_domain,
    )


def _loop_carries(
    interface: LoopInterface, nodes: tuple[Node, ...], output: Node
) -> tuple[ContractionCarry, ...] | None:
    inputs = tuple(node for node in nodes if node.op == "placeholder")
    outputs = output.args[0]
    if (
        type(interface.input_count) is not int
        or interface.input_count != len(inputs)
        or not isinstance(outputs, (tuple, list))
        or not all(isinstance(node, Node) for node in outputs)
    ):
        return None
    carried_inputs: set[int] = set()
    carried_outputs: set[int] = set()
    bindings: list[ContractionCarry] = []
    for carry in interface.carries:
        input_index, output_index = carry.input_index, carry.output_index
        if (
            type(input_index) is not int
            or type(output_index) is not int
            or not 0 <= input_index < len(inputs)
            or not 0 <= output_index < len(outputs)
            or input_index in carried_inputs
            or output_index in carried_outputs
        ):
            return None
        incoming, outgoing = inputs[input_index], outputs[output_index]
        assert isinstance(outgoing, Node)
        left, right = incoming.meta.get("val"), outgoing.meta.get("val")
        if (
            not isinstance(left, torch.Tensor)
            or not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.device != right.device
            or not _same_domain(_domain(left), _domain(right))
        ):
            return None
        carried_inputs.add(input_index)
        carried_outputs.add(output_index)
        bindings.append(ContractionCarry(input_index, output_index, incoming, outgoing))
    if carried_outputs != set(range(len(outputs))):
        return None
    return tuple(bindings)


def collect_contraction_region(info: GraphInfo) -> ContractionRegion | None:
    """Collect a single graph's contractions without imposing a workload formula.

    Mixed contraction dtypes, explicit accumulators, multiple stores, arbitrary
    coefficient producers and loop bodies are preserved. Unsupported rank or
    malformed dot/loop ports return ``None`` without changing the graph. This
    capture alone does not establish that a whole-root replacement is legal.
    """
    nodes = tuple(info.graph.nodes)
    seen: set[Node] = set()
    for node in nodes:
        if node.graph is not info.graph or any(
            source not in seen for source in node.all_input_nodes
        ):
            return None
        seen.add(node)
    outputs = tuple(node for node in nodes if node.op == "output")
    if len(outputs) != 1 or len(outputs[0].args) != 1:
        return None
    specs: list[ContractionSpec] = []
    for node in nodes:
        if node.op == "call_function" and node.target is dot:
            spec = _contraction(node)
            if spec is None:
                return None
            specs.append(spec)
    if not specs:
        return None
    live_outs: list[Node] = []

    def collect_output(node: Node) -> Node:
        live_outs.append(node)
        return node

    map_arg(outputs[0].args[0], collect_output)
    interface = info.loop_interface if isinstance(info, ForLoopGraphInfo) else None
    carries = () if interface is None else _loop_carries(interface, nodes, outputs[0])
    if carries is None:
        return None
    return ContractionRegion(
        info.graph_id,
        info.graph,
        nodes,
        tuple(specs),
        tuple(
            node
            for node in nodes
            if node.op in ("placeholder", "get_attr")
            or node.target in (_tracing_ops._host_tensor, _tracing_ops._get_symnode)
        ),
        tuple(live_outs),
        tuple(node for node in nodes if node.target is memory_ops.load),
        tuple(node for node in nodes if node.target is memory_ops.store),
        tuple(node for node in nodes if node.target is scan_ops._associative_scan),
        tuple(
            node
            for node in nodes
            if isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.reduction in node.target.tags
        ),
        interface,
        carries,
    )
