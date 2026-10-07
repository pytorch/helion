"""Prove the physical contraction axis of an interleaved packed matmul.

The ordinary matmul fact describes the logical K extent. Its expanded tile
symbol is not itself a tunable axis, so collective staging separately records
the packed axis and its expansion without changing native direct-load facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import memory_ops
from ...language.matmul_ops import MATMUL_FACT_ID_META
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from .indexing import match_cute_affine_range_iota
from .indexing import match_cute_stack_reshape_rhs

if TYPE_CHECKING:
    from torch.fx import Node

    from ...autotuner.config_spec import MatmulFact
    from ..device_ir import DeviceIR


@dataclass(frozen=True)
class PackedMatmulAxis:
    block_id: int
    factor: int


def interleaved_matrix_terms(node: Node, *, lhs: bool) -> tuple[Node, ...] | None:
    """Prove ``stack(terms, K + 1).reshape(...)`` on a two-dimensional tile."""
    value = node.meta.get("val")
    if (
        node.target
        not in (
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        )
        or len(node.args) != 2
        or node.kwargs
        or not isinstance(value, torch.Tensor)
        or value.ndim != 2
    ):
        return None
    stack = node.args[0]
    if (
        not isinstance(stack, torch.fx.Node)
        or stack.target is not torch.ops.aten.stack.default
        or len(stack.args) not in (1, 2)
        or set(stack.kwargs) - {"dim"}
    ):
        return None
    terms = stack.args[0]
    dim = stack.args[1] if len(stack.args) == 2 else stack.kwargs.get("dim", 0)
    k_dim = 1 if lhs else 0
    if (
        type(dim) is not int
        or dim not in (k_dim + 1, k_dim - 2)
        or not isinstance(terms, (tuple, list))
        or len(terms) <= 1
        or not all(isinstance(term, torch.fx.Node) for term in terms)
    ):
        return None
    proved_terms = cast("tuple[Node, ...]", tuple(terms))
    values = [term.meta.get("val") for term in proved_terms]
    first = values[0]
    if (
        not isinstance(first, torch.Tensor)
        or first.ndim != 2
        or any(
            not isinstance(term, torch.Tensor)
            or term.dtype != value.dtype
            or tuple(map(sympy.sympify, term.shape))
            != tuple(map(sympy.sympify, first.shape))
            for term in values
        )
    ):
        return None
    shape = list(map(sympy.sympify, first.shape))
    shape[k_dim] *= len(terms)
    if tuple(shape) != tuple(map(sympy.sympify, value.shape)):
        return None
    return proved_terms


def virtual_packed_terms(node: Node) -> tuple[Node, ...] | None:
    """Expose virtual elements only to matmuls whose whole axis proof succeeds."""
    env = CompileEnvironment.current()
    result = None
    if not node.users:
        return None
    for user in node.users:
        if user.target is dot:
            operands = user.args[:2]
        elif user.target is torch.ops.aten.addmm.default:
            operands = user.args[1:3]
        else:
            return None
        if len(operands) != 2 or not all(
            isinstance(arg, torch.fx.Node) for arg in operands
        ):
            return None
        lhs, rhs = operands
        assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
        if packed_matmul_axis(env, lhs, rhs) is None:
            return None
        terms = interleaved_matrix_terms(node, lhs=node is lhs)
        if terms is None or result is not None and terms != result:
            return None
        result = terms
    return result


def packed_matmul_axis(
    env: CompileEnvironment, lhs: Node, rhs: Node
) -> PackedMatmulAxis | None:
    """Require exact ``A[:, factor * tile] @ stack(B_i).reshape(...)`` order.

    Shape equalities compare symbolic expressions, never runtime hints. The
    scalar load lowering supplies the original masks and each term's casts;
    this proof only establishes the correspondence with logical K lanes.
    """
    a_value, b_value = lhs.meta.get("val"), rhs.meta.get("val")
    if (
        not isinstance(a_value, torch.Tensor)
        or not isinstance(b_value, torch.Tensor)
        or a_value.ndim != 2
        or b_value.ndim != 2
    ):
        return None
    left_terms = interleaved_matrix_terms(lhs, lhs=True)
    right_terms = interleaved_matrix_terms(rhs, lhs=False)
    if left_terms is not None and right_terms is not None:
        if len(left_terms) != len(right_terms) or a_value.dtype != b_value.dtype:
            return None
        left_k = sympy.sympify(left_terms[0].meta["val"].shape[1])
        right_k = sympy.sympify(right_terms[0].meta["val"].shape[0])
        if left_k != right_k:
            return None
        block_id = next(
            (block.block_id for block in env.block_sizes if block.var == left_k), None
        )
        return None if block_id is None else PackedMatmulAxis(block_id, len(left_terms))
    if lhs.target is not memory_ops.load or len(lhs.args) < 2:
        return None
    subscript = lhs.args[1]
    if not isinstance(subscript, (tuple, list)) or len(subscript) != 2:
        return None
    index = subscript[1]
    if not isinstance(index, torch.fx.Node):
        return None
    affine = match_cute_affine_range_iota(index)
    packed = match_cute_stack_reshape_rhs(rhs)
    if affine is None or packed is None or not isinstance(affine.base, torch.fx.Node):
        return None
    terms, factor = packed
    if factor != affine.factor:
        return None
    base = affine.base.meta.get("val")
    if not isinstance(base, (int, torch.SymInt)):
        return None
    base_expr = sympy.sympify(base)
    block_id = next(
        (block.block_id for block in env.block_sizes if block.var == base_expr), None
    )
    if block_id is None:
        return None
    shape = (base_expr, sympy.sympify(b_value.shape[1]))
    if (
        sympy.sympify(a_value.shape[1]) != factor * shape[0]
        or sympy.sympify(b_value.shape[0]) != factor * shape[0]
    ):
        return None
    for term in terms:
        value = term.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.dtype != b_value.dtype
            or tuple(map(sympy.sympify, value.shape)) != shape
        ):
            return None
    return PackedMatmulAxis(block_id, factor)


def packed_axis_for_fact(
    env: CompileEnvironment, device_ir: DeviceIR, fact: MatmulFact
) -> PackedMatmulAxis | None:
    """Recover an unambiguous packed axis from the fact's attributed FX nodes."""
    fact_ids = {
        index
        for index, value in enumerate(env.config_spec.matmul_facts)
        if value == fact
    }
    result = None
    for graph in device_ir.graphs:
        for node in graph.graph.nodes:
            if node.meta.get(MATMUL_FACT_ID_META) not in fact_ids:
                continue
            if node.target is dot:
                operands = node.args[:2]
            elif node.target is torch.ops.aten.addmm.default:
                operands = node.args[1:3]
            else:
                return None
            if len(operands) != 2 or not all(
                isinstance(operand, torch.fx.Node) for operand in operands
            ):
                return None
            lhs, rhs = operands
            assert isinstance(lhs, torch.fx.Node)
            assert isinstance(rhs, torch.fx.Node)
            axis = packed_matmul_axis(env, lhs, rhs)
            if axis is None or result is not None and result != axis:
                return None
            result = axis
    return result
