"""A bounded typed scalar-only prefix of one FP32 per-M load.

This proof is invoked only after an explicit singleton view owns the M axis.
It is not a generic vector/broadcast admission, nor a readonly/alias proof.
Original typed operations form an ordered DAG and retain scalar storage rounding,
cast points and binary association; no arithmetic is simplified or reassociated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .cute_epilogue import _AuxiliaryTensorLoadExpr
    from .cute_epilogue import _PointwiseDagExpr


@dataclass(frozen=True)
class MVectorScalarPrefix:
    load_node: torch.fx.Node
    raw_expr: _AuxiliaryTensorLoadExpr
    dag: _PointwiseDagExpr
    owned_nodes: tuple[torch.fx.Node, ...]
    depth: int


def prove_m_vector_scalar_prefix(
    node: torch.fx.Node,
    shape: tuple[object, ...],
    device: torch.device,
) -> MVectorScalarPrefix | None:
    """Only FP32 arithmetic with one M load and finite, typed FP32 literals.

    FP16/BF16 casts of the vector are explicit operations; arithmetic after a
    narrowing cast needs an explicit return to FP32. Raw half/BOOL/other vectors,
    rank-zero runtime tensors, two vectors and scalar expression DAGs remain
    unsupported. The existing direct-load/casts path owns its unchanged scope.
    """
    from ...language import memory_ops
    from . import cute_epilogue as epi

    memo: dict[torch.fx.Node, epi._TensorExpr] = {}
    depths: dict[torch.fx.Node, int] = {}
    active: set[torch.fx.Node] = set()
    ordered: list[epi._TensorExpr] = []
    provenance: list[tuple[torch.fx.Node, torch.dtype]] = []
    load: torch.fx.Node | None = None
    raw: epi._AuxiliaryTensorLoadExpr | None = None
    arithmetic = False

    def parse(current: torch.fx.Node, depth: int) -> epi._TensorExpr | None:
        nonlocal load, raw, arithmetic
        if current in memo:
            return memo[current]
        if (
            depth > 32
            or len(memo) + len(active) >= 64
            or current in active
            or current.graph is not node.graph
            or current.op != "call_function"
            or current.kwargs
        ):
            return None
        value = current.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.device != device:
            return None
        node_shape = tuple(value.shape)
        if node_shape not in (shape, ()):
            return None
        active.add(current)
        expr: epi._TensorExpr | None = None
        children: list[torch.fx.Node] = []
        if node_shape == ():
            constant = epi._scalar_constant(current)
            if constant is not None and constant.dtype is torch.float32:
                expr = constant
        elif current.target is memory_ops.load:
            if load is None and value.dtype is torch.float32:
                load = current
                raw = epi._AuxiliaryTensorLoadExpr(current, None, "{aux}")
                expr = raw
        elif (cast := epi._floating_cast_step(current)) is not None:
            step, operand = cast
            before = operand.meta.get("val")
            if (
                isinstance(before, torch.Tensor)
                and tuple(before.shape) == shape
                and before.device == device
                and value.dtype in epi._FLOAT_CAST_TYPES
            ):
                child = parse(operand, depth + 1)
                if child is not None:
                    expr = epi._UnaryTensorExpr(step, child)
                    children.append(operand)
        elif current.target in (torch.ops.aten.neg.default, torch.ops.aten.abs.default):
            if len(current.args) == 1 and isinstance(current.args[0], torch.fx.Node):
                operand = current.args[0]
                before = operand.meta.get("val")
                step = epi._unary_step_for(current.target)
                if (
                    isinstance(before, torch.Tensor)
                    and before.dtype is torch.float32
                    and value.dtype is torch.float32
                    and tuple(before.shape) == shape
                    and step is not None
                ):
                    child = parse(operand, depth + 1)
                    if child is not None:
                        expr = epi._UnaryTensorExpr(step, child)
                        children.append(operand)
                        arithmetic = True
        elif (
            current.target in epi._AUX_EXPR_BINARY_TEMPLATES and len(current.args) == 2
        ):
            lhs, rhs = current.args
            tensor_args = [
                arg
                for arg in (lhs, rhs)
                if isinstance(arg, torch.fx.Node)
                and isinstance(arg.meta.get("val"), torch.Tensor)
                and tuple(arg.meta["val"].shape) == shape
            ]
            if len(tensor_args) == 1 and value.dtype is torch.float32:
                vector = tensor_args[0]
                vector_value = vector.meta.get("val")
                child = parse(vector, depth + 1)
                if (
                    child is not None
                    and isinstance(vector_value, torch.Tensor)
                    and vector_value.dtype is torch.float32
                    and epi._operand_dtype_matches(child, vector, torch.float32)
                ):
                    other = rhs if vector is lhs else lhs
                    if isinstance(other, torch.fx.Node):
                        constant = parse(other, depth + 1)
                        if (
                            isinstance(constant, epi._ScalarConstantExpr)
                            and constant.dtype is torch.float32
                        ):
                            children.extend((vector, other))
                            expr = epi._BinaryTensorExpr(
                                epi._BINARY_OP_NAMES[current.target],
                                epi._round_epilogue_expression(
                                    epi._AUX_EXPR_BINARY_TEMPLATES[current.target],
                                    torch.float32,
                                ),
                                child if vector is lhs else constant,
                                constant if vector is lhs else child,
                            )
                    else:
                        # Weak Python scalars are converted to the FP32 vector's
                        # arithmetic dtype. Reject large integers before float()
                        # can overflow or double-round, as typed full() does.
                        scalar_in_range = (
                            isinstance(other, (int, float))
                            and not isinstance(other, bool)
                            and (not isinstance(other, int) or abs(other) <= 1 << 24)
                            and abs(other) <= torch.finfo(torch.float32).max
                        )
                        literal = (
                            epi._extract_scalar(other) if scalar_in_range else None
                        )
                        if literal is not None:
                            constant = epi._ScalarConstantExpr(
                                repr(literal), torch.float32
                            )
                            ordered.append(constant)
                            provenance.append((current, torch.float32))
                            expr = epi._BinaryTensorExpr(
                                epi._BINARY_OP_NAMES[current.target],
                                epi._round_epilogue_expression(
                                    epi._AUX_EXPR_BINARY_TEMPLATES[current.target],
                                    torch.float32,
                                ),
                                child if vector is lhs else constant,
                                constant if vector is lhs else child,
                            )
                            children.append(vector)
                    if expr is not None:
                        arithmetic = True
        active.remove(current)
        if expr is None:
            return None
        depths[current] = 1 + max((depths[child] for child in children), default=0)
        memo[current] = expr
        ordered.append(expr)
        provenance.append((current, value.dtype))
        return expr

    root = parse(node, 0)
    if (
        root is None
        or load is None
        or raw is None
        or not arithmetic
        or len(ordered) > 64
    ):
        return None
    return MVectorScalarPrefix(
        load,
        raw,
        epi._PointwiseDagExpr(tuple(ordered), tuple(provenance)),
        tuple(memo),
        depths[node],
    )
