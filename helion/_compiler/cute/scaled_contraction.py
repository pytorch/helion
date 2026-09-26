"""Expose exactly representable scaled low-precision contractions to MMA.

A decoded E2M1 value is an integer of magnitude at most 12 divided by two.
Multiplying it by an E4M3FN scale requires at most eight significant bits and
fits in BF16's normal range. A sum of two E2M1 products requires at most nine
bits; applying two E4M3FN scales requires at most seventeen, so distributing
those scales is exact in FP32. No arbitrary FP32 tensor is narrowed here.

The pass accepts a positive-zero FP32 matrix carry and explicit broadcasts,
pair sums and reductions. It preserves FP32 accumulation, with the ordinary
GEMM freedom to reorder its reduction. The positive-zero carry normalizes the
signed zeros that scale distribution can change. E4M3FN has no infinities;
NaNs still propagate. Padding is zeroed *after* scaling and before interleaving
so an inactive K lane cannot introduce a scale NaN into the contraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
from ...language.matmul_ops import MATMUL_FACT_ID_META
from ...language.matmul_ops import dot
from ...language.matmul_ops import enforce_dot_requirements
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import control_flow_parent_entries
from .block_scaled_provenance import BLOCK_SCALED_PROVENANCE
from .block_scaled_provenance import capture_block_scaled_provenance
from .fold_noop_stores import _is_read_only

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx import Node
    from torch.fx.node import Argument

    from ..device_ir import DeviceIR

_ADD = torch.ops.aten.add.Tensor
_MUL = torch.ops.aten.mul.Tensor
_SUM = torch.ops.aten.sum.dim_IntList
_CONVERT = torch.ops.prims.convert_element_type.default


def _value(node: object, rank: int, dtype: torch.dtype) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != rank
        or value.dtype != dtype
    ):
        return None
    return value


def _same_shape(left: torch.Tensor, right: torch.Tensor) -> bool:
    return tuple(map(sympy.sympify, left.shape)) == tuple(
        map(sympy.sympify, right.shape)
    )


def _binary(node: object, target: object) -> tuple[Node, Node] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not target
        or len(node.args) != 2
        or node.kwargs
        or not all(isinstance(arg, torch.fx.Node) for arg in node.args)
    ):
        return None
    return cast("tuple[Node, Node]", node.args)


def _unbroadcast(node: Node, dim: int) -> Node | None:
    if (
        node.target is not torch.ops.aten.unsqueeze.default
        or len(node.args) != 2
        or node.kwargs
        or node.args[1] not in (dim, dim - 3)
        or _value(node, 3, torch.float32) is None
        or _value(node.args[0], 2, torch.float32) is None
    ):
        return None
    return cast("Node", node.args[0])


def _fp4_value(node: Node) -> bool:
    if (
        node.target is not operator.getitem
        or len(node.args) != 2
        or node.kwargs
        or type(node.args[1]) is not int
        or node.args[1] not in (0, 1)
        or _value(node, 2, torch.float32) is None
    ):
        return False
    pair = node.args[0]
    return (
        isinstance(pair, torch.fx.Node)
        and pair.target is float4_e2m1fn_x2_to_float32
        and len(pair.args) == 1
        and not pair.kwargs
        and _value(pair.args[0], 2, torch.float4_e2m1fn_x2) is not None
    )


def _fp8_scale(node: Node) -> bool:
    return (
        node.target is _CONVERT
        and len(node.args) == 2
        and not node.kwargs
        and node.args[1] is torch.float32
        and _value(node.args[0], 2, torch.float8_e4m3fn) is not None
        and _value(node, 2, torch.float32) is not None
    )


@dataclass(frozen=True)
class _ScaledPair:
    lhs: tuple[Node, Node]
    rhs: tuple[Node, Node]
    lhs_scale: Node
    rhs_scale: Node


def _product(node: Node) -> tuple[Node, Node] | None:
    operands = _binary(node, _MUL)
    if operands is None or _value(node, 3, torch.float32) is None:
        return None
    for left, right in (operands, operands[::-1]):
        lhs, rhs = _unbroadcast(left, 2), _unbroadcast(right, 0)
        if lhs is not None and rhs is not None and _fp4_value(lhs) and _fp4_value(rhs):
            return lhs, rhs
    return None


def _scaled_pair(node: Node) -> _ScaledPair | None:
    """Match ``sum((a0*b0 + a1*b1) * sa * sb, dim=1)`` exactly."""
    if (
        node.target is not _SUM
        or len(node.args) not in (2, 3)
        or node.args[1] not in ([1], [-2], (1,), (-2,))
        or len(node.args) == 3
        and node.args[2] is not False
        or set(node.kwargs) - {"keepdim", "dtype"}
        or node.kwargs.get("keepdim", False) is not False
        or node.kwargs.get("dtype") not in (None, torch.float32)
        or _value(node, 2, torch.float32) is None
    ):
        return None
    outer = _binary(node.args[0], _MUL)
    if outer is None:
        return None
    for inner_node, scale_outer in (outer, outer[::-1]):
        inner = _binary(inner_node, _MUL)
        if inner is None:
            continue
        for pair_node, scale_inner in (inner, inner[::-1]):
            pairs = _binary(pair_node, _ADD)
            if pairs is None:
                continue
            first, second = (_product(product) for product in pairs)
            if first is None or second is None:
                continue
            for a_broadcast, b_broadcast in (
                (scale_outer, scale_inner),
                (scale_inner, scale_outer),
            ):
                sa, sb = _unbroadcast(a_broadcast, 2), _unbroadcast(b_broadcast, 0)
                if sa is None or sb is None or not _fp8_scale(sa) or not _fp8_scale(sb):
                    continue
                a_value, b_value = first[0].meta["val"], first[1].meta["val"]
                output = node.meta["val"]
                if (
                    not all(
                        _same_shape(term.meta["val"], a_value)
                        for term in (second[0], sa)
                    )
                    or not all(
                        _same_shape(term.meta["val"], b_value)
                        for term in (second[1], sb)
                    )
                    or sympy.sympify(a_value.shape[1])
                    != sympy.sympify(b_value.shape[0])
                    or tuple(map(sympy.sympify, output.shape))
                    != (
                        sympy.sympify(a_value.shape[0]),
                        sympy.sympify(b_value.shape[1]),
                    )
                ):
                    continue
                return _ScaledPair((first[0], second[0]), (first[1], second[1]), sa, sb)
    return None


def _zero_carry(device_ir: DeviceIR, info: ForLoopGraphInfo, node: Node) -> bool:
    parent_entry = control_flow_parent_entries(device_ir.graphs).get(info.graph_id)
    if parent_entry is None:
        return False
    parent, slot = parent_entry
    captures = parent.args[slot]
    if not isinstance(captures, (tuple, list)) or len(captures) != 1:
        return False
    seed = captures[0]
    if (
        not isinstance(seed, torch.fx.Node)
        or seed.target is not creation_ops.full
        or len(seed.args) < 2
        or _value(seed, 2, torch.float32) is None
        or not _same_shape(seed.meta["val"], node.meta["val"])
    ):
        return False
    value = seed.args[1]
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
        and math.copysign(1, value) > 0
    )


def _rewrite_loop(device_ir: DeviceIR, info: ForLoopGraphInfo) -> bool:
    env = CompileEnvironment.current()
    if len(info.block_ids) != 1 or len(info.node_args) != 1:
        return False
    graph = info.graph
    placeholders = list(graph.find_nodes(op="placeholder"))
    outputs = list(graph.find_nodes(op="output"))
    if len(placeholders) != 1 or len(outputs) != 1:
        return False
    output = outputs[0]
    if (
        len(output.args) != 1
        or not isinstance(output.args[0], (list, tuple))
        or len(output.args[0]) != 1
    ):
        return False
    result = output.args[0][0]
    if (
        not isinstance(result, torch.fx.Node)
        or _value(result, 2, torch.float32) is None
    ):
        return False
    carry = placeholders[0]
    if not _zero_carry(device_ir, info, carry):
        return False
    current = result
    pairs: list[_ScaledPair] = []
    while (args := _binary(current, _ADD)) is not None:
        previous, reduction = args
        if tuple(current.users) != (output,) and len(current.users) != 1:
            return False
        pair = _scaled_pair(reduction)
        if pair is None or tuple(reduction.users) != (current,):
            return False
        pairs.append(pair)
        current = previous
    if not pairs:
        return False
    accumulator = current
    if (
        current.target is _tracing_ops._new_var
        and len(current.args) == 1
        and not current.kwargs
    ):
        current = current.args[0]
    if current is not carry or len(accumulator.users) != 1:
        return False
    if any(
        user is not accumulator and user.target is not torch.ops.aten.sym_size.int
        for user in carry.users
    ):
        return False
    if any(
        node.op != "output"
        and node is not accumulator
        and node.target is not float4_e2m1fn_x2_to_float32
        and not _is_read_only(node)
        for node in graph.nodes
    ):
        return False
    pairs.reverse()
    left_value = pairs[0].lhs[0].meta["val"]
    right_value = pairs[0].rhs[0].meta["val"]
    k_expr = sympy.sympify(left_value.shape[1])
    k_id = env.canonical_block_id(info.block_ids[0])
    if (
        k_expr != env.block_sizes[k_id].var
        or k_expr in map(sympy.sympify, result.meta["val"].shape)
        or any(
            not _same_shape(pair.lhs[0].meta["val"], left_value)
            or not _same_shape(pair.rhs[0].meta["val"], right_value)
            for pair in pairs[1:]
        )
    ):
        return False
    old_dependencies: set[Node] = set()
    pending = [result]
    while pending:
        node = pending.pop()
        if node not in old_dependencies:
            old_dependencies.add(node)
            pending.extend(node.all_input_nodes)

    def call(
        target: Callable[..., object], args: tuple[Argument, ...], value: object
    ) -> Node:
        new = graph.call_function(target, args)
        new.meta = {**result.meta, "val": value}
        for key in (
            "lowering",
            "masked_value",
            MATMUL_FACT_ID_META,
            MATMUL_DIM_BLOCK_IDS_META,
        ):
            new.meta.pop(key, None)
        return new

    with graph.inserting_before(result):
        lhs_terms, rhs_terms = [], []
        for pair in pairs:
            for terms, source, scale in (
                (lhs_terms, pair.lhs, pair.lhs_scale),
                (rhs_terms, pair.rhs, pair.rhs_scale),
            ):
                for term in source:
                    value = term.meta["val"]
                    product = call(_MUL, (term, scale), value.new_empty(value.shape))
                    # The byte load's zero fill does not prove a zero product:
                    # an independently indexed scale can contain a padding NaN.
                    # The scalar masked-value lattice otherwise folds 0*x to 0,
                    # discarding the mask needed before K is interleaved.
                    product.meta["masked_value"] = None
                    narrowed_value = value.new_empty(value.shape, dtype=torch.bfloat16)
                    narrowed = call(_CONVERT, (product, torch.bfloat16), narrowed_value)
                    terms.append(
                        call(_tracing_ops._mask_to, (narrowed, 0), narrowed_value)
                    )

        factor = len(lhs_terms)

        def interleave(terms: list[Node], dim: int) -> Node:
            value = terms[0].meta["val"]
            shape = list(value.shape)
            shape.insert(dim, factor)
            stack = call(
                torch.ops.aten.stack.default, (terms, dim), value.new_empty(shape)
            )
            logical_shape = list(value.shape)
            logical_shape[dim - 1] *= factor
            sizes = [
                call(torch.ops.aten.sym_size.int, (terms[0], axis), size)
                for axis, size in enumerate(value.shape)
            ]
            sizes[dim - 1] = call(
                operator.mul, (sizes[dim - 1], factor), logical_shape[dim - 1]
            )
            return call(
                torch.ops.aten.reshape.default,
                (stack, sizes),
                value.new_empty(logical_shape),
            )

        lhs, rhs = interleave(lhs_terms, 2), interleave(rhs_terms, 1)
        contraction = call(
            dot, (lhs, rhs, accumulator, torch.float32), result.meta["val"]
        )
        fact_id = enforce_dot_requirements(lhs.meta["val"], rhs.meta["val"])
        fact = env.config_spec.matmul_facts[fact_id]
        problem_k = env.block_sizes[k_id].numel
        env.config_spec.matmul_facts[fact_id] = fact._replace(
            static_k=int(problem_k) * factor if problem_k.is_Integer else None,
        )
        contraction.meta[MATMUL_FACT_ID_META] = fact_id
        contraction.meta[MATMUL_DIM_BLOCK_IDS_META] = (fact.m_block_id, fact.n_block_id)
        if fact.m_block_id is not None and fact.n_block_id is not None:
            provenance = capture_block_scaled_provenance(
                pairs,
                lowered_operands=(lhs, rhs),
                group_axis=k_id,
                row_axes=(fact.m_block_id, fact.n_block_id),
                loop_graph_id=info.graph_id,
                fact_id=fact_id,
            )
            if provenance is not None:
                contraction.meta[BLOCK_SCALED_PROVENANCE] = provenance
                env.config_spec.cute_scaled_mma_available = True
    result.replace_all_uses_with(contraction)
    for node in reversed(list(graph.nodes)):
        if (
            node in old_dependencies
            and not node.users
            and node.op == "call_function"
            and (_is_read_only(node) or node.target is float4_e2m1fn_x2_to_float32)
        ):
            graph.erase_node(node)
    graph.lint()
    return True


def expose_scaled_contractions(device_ir: DeviceIR) -> int:
    """Rewrite proved loop carries before ordinary lowering preparation."""
    if CompileEnvironment.current().backend_name != "cute":
        return 0
    return sum(
        _rewrite_loop(device_ir, info)
        for info in device_ir.graphs
        if isinstance(info, ForLoopGraphInfo)
    )
